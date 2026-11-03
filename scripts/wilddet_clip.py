#!/usr/bin/env python3
"""WildDet3D on one front-camera video.

Writes one JSON the overlay can draw. Each detection keeps the prompt name,
the 2D box, the camera-frame centre and that centre moved into the vehicle
frame. The 3D centre is the model's. This script does not use the ground plane.

The camera JSON is the same file the overlay uses. WildDet3D reads fx, fy, cx,
cy as a pinhole. A CylinderV01 view is not a pinhole, so the range is off away
from the image centre. The class name does not depend on that.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

PROMPTS = ["car", "bus", "truck", "train", "bicycle", "person"]


def as_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def optical_to_ego(center, position):
    """Camera optical axes (x right, y down, z forward) to the vehicle frame."""
    x, y, z = (float(v) for v in center)
    return [
        round(z + float(position[0]), 3),
        round(-x + float(position[1]), 3),
        round(-y + float(position[2]), 3),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/wilddet3d_alldata_all_prompt_v1.0.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score", type=float, default=0.2)
    parser.add_argument("--repeat", type=int, default=1, help="Video frames per kept frame. Use 2 when each frame is stored twice.")
    parser.add_argument("--prompts", nargs="+", default=PROMPTS)
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    camera = json.loads(args.camera.read_text())
    intrinsic = np.array(
        [[camera["fx"], 0.0, camera["cx"]], [0.0, camera["fy"], camera["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    position = camera["position_m"]

    sys.path.insert(0, str(Path("third_party/WildDet3D").resolve()))
    from wilddet3d import build_model, preprocess

    print("building WildDet3D", flush=True)
    model = build_model(
        checkpoint=str(args.checkpoint),
        score_threshold=args.score,
        score_3d_threshold=0.05,
        skip_pretrained=True,
        device=args.device,
    )
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")

    frames_out = []
    kept = 0
    seen = 0
    while True:
        ok, image_bgr = cap.read()
        if not ok:
            break
        if seen % args.repeat != 0:
            seen += 1
            continue
        seen += 1
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        data = preprocess(image, intrinsic)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            results = model(
                images=data["images"].to(args.device),
                intrinsics=data["intrinsics"].to(args.device)[None],
                input_hw=[data["input_hw"]],
                original_hw=[data["original_hw"]],
                padding=[data["padding"]],
                input_texts=list(args.prompts),
            )
        boxes2d = as_numpy(results[0][0]) if len(results) > 0 else None
        boxes3d = as_numpy(results[1][0]) if len(results) > 1 else None
        scores = as_numpy(results[2][0]) if len(results) > 2 else None
        class_ids = as_numpy(results[5][0]) if len(results) > 5 else None
        detections = []
        n = 0 if boxes3d is None else len(boxes3d)
        for i in range(n):
            box = np.asarray(boxes3d[i]).reshape(-1)
            score = 1.0 if scores is None else float(scores[i])
            if score < args.score or box.shape[0] < 3 or not np.isfinite(box[:3]).all():
                continue
            cid = -1 if class_ids is None else int(class_ids[i])
            name = args.prompts[cid] if 0 <= cid < len(args.prompts) else f"id{cid}"
            xyxy = None
            if boxes2d is not None and i < len(boxes2d):
                xyxy = [round(float(v), 1) for v in np.asarray(boxes2d[i]).reshape(-1)[:4]]
            detections.append(
                {
                    "name": name,
                    "score": round(score, 4),
                    "xyxy": xyxy,
                    "center_cam": [round(float(v), 3) for v in box[:3]],
                    "center_ego": optical_to_ego(box[:3], position),
                    "box_cam": [round(float(v), 4) for v in box],
                }
            )
        kept += len(detections)
        frames_out.append({"detections": detections})
        if len(frames_out) == 1:
            shape = None if boxes3d is None else list(np.shape(boxes3d))
            print("first frame", shape, "kept", len(detections), flush=True)
        if len(frames_out) % 15 == 0:
            print(len(frames_out), flush=True)
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "WildDet3D",
        "prompts": list(args.prompts),
        "repeat": args.repeat,
        "frames": frames_out,
    }
    args.output.write_text(json.dumps(payload))
    summary = [
        "# WildDet3D on one clip",
        "",
        "- class id is the index into the prompt list",
        "- centre is the model output, then camera optical axes to the vehicle frame",
        "- the intrinsic is passed as a pinhole",
        f"- prompts: {', '.join(args.prompts)}",
        f"- frames: {len(frames_out)}",
        f"- boxes: {kept}",
    ]
    args.output.with_suffix(".md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
