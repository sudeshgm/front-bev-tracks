#!/usr/bin/env python3
"""WildDet3D on the front camera. The 3D centre comes from the model, not the ground plane."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

PROMPTS = ["car", "bus", "truck"]


def as_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/wilddet3d_alldata_all_prompt_v1.0.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score", type=float, default=0.2)
    args = parser.parse_args()

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
    folders = sorted(path for path in args.frames.iterdir() if (path / "frame.json").exists())
    if not folders:
        raise SystemExit(f"no frames in {args.frames}")

    args.output.mkdir(parents=True, exist_ok=True)
    frames_out = []
    kept = 0
    ranges = []
    for index, folder in enumerate(folders):
        meta = json.loads((folder / "frame.json").read_text())
        calib = np.load(folder / "calib.npz")
        image_bgr = cv2.imread(str(folder / "images" / "CAM_FRONT.jpg"))
        if image_bgr is None:
            raise SystemExit(f"missing front image in {folder}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        intrinsic = calib["cam_intrinsic"][0].astype(np.float64)
        rotation = calib["sensor2lidar_rotation"][0].astype(np.float64)
        translation = calib["sensor2lidar_translation"][0].astype(np.float64)
        data = preprocess(image, intrinsic)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                results = model(
                    images=data["images"].to(args.device),
                    intrinsics=data["intrinsics"].to(args.device)[None],
                    input_hw=[data["input_hw"]],
                    original_hw=[data["original_hw"]],
                    padding=[data["padding"]],
                    input_texts=PROMPTS,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise SystemExit(
                "WildDet3D ran out of GPU memory on the first frames. "
                "The checkpoint is 4.7 GB and this machine is a 16 GB T4."
            )
        boxes3d = as_numpy(results[1][0]) if len(results) > 1 else None
        scores = as_numpy(results[2][0]) if len(results) > 2 else None
        detections = []
        if boxes3d is not None and len(boxes3d):
            if boxes3d.ndim == 1:
                boxes3d = boxes3d.reshape(1, -1)
            if scores is None:
                scores = np.ones(len(boxes3d), dtype=np.float32)
            for box, score in zip(boxes3d, scores):
                if float(score) < args.score or box.shape[0] < 7:
                    continue
                centre = rotation @ box[:3].astype(np.float64) + translation
                if not np.isfinite(centre).all():
                    continue
                detections.append(
                    {
                        "name": "vehicle",
                        "score": round(float(score), 4),
                        "center_ego": [round(float(v), 3) for v in centre],
                        "box_cam": [round(float(v), 4) for v in box[:7]],
                    }
                )
                ranges.append(float(centre[0]))
        kept += len(detections)
        frames_out.append({"token": meta["token"], "detections": detections})
        if index == 0:
            shape = None if boxes3d is None else list(boxes3d.shape)
            print("first frame boxes3d", shape, "kept", len(detections), flush=True)
        if index % 15 == 0:
            print(index, flush=True)
        torch.cuda.empty_cache()

    payload = {
        "model": "WildDet3D",
        "checkpoint": args.checkpoint.name,
        "prompt": PROMPTS,
        "camera": "CAM_FRONT",
        "frames": frames_out,
    }
    (args.output / "eval.json").write_text(json.dumps(payload))
    finite = [v for v in ranges if np.isfinite(v)]
    summary = [
        "# WildDet3D on the Gaimersheim front camera",
        "",
        "- 3D centre from the model, then the Audi extrinsics",
        "- no ground plane, no cuboid guess",
        f"- prompts: {', '.join(PROMPTS)}",
        f"- frames: {len(frames_out)}",
        f"- boxes: {kept}",
        f"- range m, median: {round(float(np.median(finite)), 1) if finite else 'none'}",
    ]
    (args.output / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
