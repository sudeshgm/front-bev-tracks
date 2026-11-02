#!/usr/bin/env python3
"""Anchor3DLane on the front camera.

The 3D points come from the model. They are not dropped onto the ground plane.
The projection matrix is this rig's, so the anchors are sampled in the Audi image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdtrack.geometry import build_lidar2img  # noqa: E402

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
INPUT_H, INPUT_W = 360, 480
Y_STEPS = np.array(
    [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    dtype=np.float32,
)


def projection(intrinsic: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground-to-image matrix in OpenLane axes, and the camera origin in ego.

    OpenLane ground is x right, y forward, z up, with the origin under the camera.
    Ego is x forward, y left, z up.
    """
    ego = build_lidar2img(intrinsic, rotation, translation)[:3]
    origin = np.asarray(translation, dtype=np.float64)
    ground_to_ego = np.array(
        [
            [0.0, 1.0, 0.0, origin[0]],
            [-1.0, 0.0, 0.0, origin[1]],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ego @ ground_to_ego, origin


def prepare(image: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(image, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def decode(proposals: torch.Tensor, scores: torch.Tensor, origin: np.ndarray) -> list[dict]:
    lanes = []
    length = len(Y_STEPS)
    for lane, score in zip(proposals, scores):
        lane = lane.detach().cpu().numpy()
        visible = lane[5 + 2 * length : 5 + 3 * length] > 0.5
        if int(visible.sum()) < 2:
            continue
        right = lane[5 : 5 + length][visible]
        forward = Y_STEPS[visible]
        height = lane[5 + length : 5 + 2 * length][visible]
        ego = np.stack([origin[0] + forward, origin[1] - right, height], axis=1)
        logits = lane[5 + 3 * length :]
        category = int(np.argmax(logits[1:])) + 1 if logits.shape[0] > 1 else 0
        lanes.append(
            {
                "score": round(float(score), 4),
                "category": category,
                "ego": np.round(ego, 3).tolist(),
            }
        )
    return lanes


def gaps_at(frames: list[dict], forward: float) -> list[float]:
    found = []
    for frame in frames:
        lateral = []
        for lane in frame["lanes"]:
            pts = [p for p in lane["ego"] if abs(p[0] - forward) < 3]
            if pts:
                lateral.append(float(np.median([p[1] for p in pts])))
        lateral.sort()
        found.extend(b - a for a, b in zip(lateral, lateral[1:]) if b - a > 0.4)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=ROOT / "data" / "frames")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score", type=float, default=0.2)
    args = parser.parse_args()

    from mmcv import Config
    from mmseg.models import build_lanedetector

    cfg = Config.fromfile(str(args.config))
    cfg.model.pretrained = None
    cfg.model.backbone.pretrained = None
    model = build_lanedetector(cfg.model)
    blob = torch.load(args.checkpoint, map_location="cpu")
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    state = {key.replace("module.", ""): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing {len(missing)} unexpected {len(unexpected)}", flush=True)
    if len(missing) > 20:
        raise SystemExit(f"checkpoint did not match the model, first missing: {missing[:8]}")
    model.to(args.device).eval()

    folders = sorted(p for p in args.frames.iterdir() if (p / "images" / "CAM_FRONT.jpg").exists())
    if not folders:
        raise SystemExit(f"no frames in {args.frames}")
    sample = cv2.imread(str(folders[0] / "images" / "CAM_FRONT.jpg"))
    height, width = sample.shape[:2]
    model.db_cfg.org_h = height
    model.db_cfg.org_w = width
    print(f"image {width}x{height}  db org {model.db_cfg.org_w}x{model.db_cfg.org_h}", flush=True)

    length = len(Y_STEPS)
    frames_out = []
    kept = 0
    for index, folder in enumerate(folders):
        image = cv2.imread(str(folder / "images" / "CAM_FRONT.jpg"))
        calib = np.load(folder / "calib.npz")
        matrix, origin = projection(
            calib["cam_intrinsic"][0].astype(np.float64),
            calib["sensor2lidar_rotation"][0].astype(np.float64),
            calib["sensor2lidar_translation"][0].astype(np.float64),
        )
        tensor = prepare(image).unsqueeze(0).to(args.device)
        mask = torch.zeros(1, 1, INPUT_H, INPUT_W, device=args.device)
        project = torch.from_numpy(matrix).float().unsqueeze(0).to(args.device)
        with torch.no_grad():
            output, _aux = model.encoder_decoder(tensor, mask, project)
            proposals = output["reg_proposals"][0]
            logits = proposals[:, 5 + 3 * length : 5 + 3 * length + model.num_category]
            scores = 1 - F.softmax(logits, dim=1)[:, 0]
            keep = scores > args.score
            lanes = decode(proposals[keep], scores[keep], origin)
        kept += len(lanes)
        meta = json.loads((folder / "frame.json").read_text())
        frames_out.append({"token": meta["token"], "lanes": lanes})
        if index % 15 == 0:
            print(f"{index} lanes {len(lanes)}", flush=True)

    near = gaps_at(frames_out, origin[0] + 10)
    far = gaps_at(frames_out, origin[0] + 30)
    summary = [
        "# Anchor3DLane on the Gaimersheim front camera",
        "",
        f"- frames: {len(frames_out)}",
        f"- lanes: {kept}",
        f"- score floor: {args.score}",
        "- 3D points are the model output, converted with this camera's extrinsics",
        f"- adjacent gap at 10 m ahead: median {float(np.median(near)):.2f} m over {len(near)} pairs" if near else "- adjacent gap at 10 m ahead: none",
        f"- adjacent gap at 30 m ahead: median {float(np.median(far)):.2f} m over {len(far)} pairs" if far else "- adjacent gap at 30 m ahead: none",
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "Anchor3DLane",
        "checkpoint": "openlane_anchor3dlane.pth",
        "camera": "CAM_FRONT",
        "score_min": args.score,
        "frames": frames_out,
    }
    (args.output / "eval.json").write_text(json.dumps(payload))
    (args.output / "summary.md").write_text("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)


if __name__ == "__main__":
    main()
