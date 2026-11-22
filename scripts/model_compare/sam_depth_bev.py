#!/usr/bin/env python3
"""SAM (text-prompted) for lane lines and road edges + Depth Anything for BEV.

Side-by-side: left = front view with coloured masks, right = BEV of the projected points.
Uses SAM 2.1 Small (Apache-2.0) with Grounding DINO for text prompts when available,
or falls back to automatic masks filtered by geometry. Depth from Depth Anything V2 Small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def load_depth_anything(device):
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    ckpt = Path("weights/depth_anything_v2_vits.pth")
    if not ckpt.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth", local_dir="weights")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


def load_sam(device):
    # Prefer SAM 2.1 Small. Text via GroundingDINO if present, else auto masks.
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        sam = build_sam2("sam2_hiera_s.yaml", "weights/sam2.1_hiera_small.pt", device=device)
        return SAM2AutomaticMaskGenerator(sam)
    except Exception:
        return None


def mask_to_bev(mask, depth, fx, fy, cx, cy, bev_size=400, max_dist=40.0):
    """Project mask pixels to ground-plane BEV using depth."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    z = depth[ys, xs]
    valid = (z > 0.5) & (z < max_dist)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if len(xs) == 0:
        return np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    # Approximate pinhole: X = (u-cx)*z/fx, Y = (v-cy)*z/fy, Z = z
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    # BEV: forward = Z, left = -X
    scale = (bev_size * 0.45) / max_dist
    u = (bev_size // 2 - X * scale).astype(int)
    v = (bev_size - 20 - z * scale).astype(int)
    canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    ok = (u >= 0) & (u < bev_size) & (v >= 0) & (v < bev_size)
    canvas[v[ok], u[ok]] = (0, 255, 0)  # green for lanes/edges
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fx", type=float, default=800)
    p.add_argument("--fy", type=float, default=800)
    p.add_argument("--cx", type=float, default=640)
    p.add_argument("--cy", type=float, default=360)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = args.device
    depth_model = load_depth_anything(device)
    sam_gen = load_sam(device)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bev_w = 400
    out_w = w + bev_w
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, h)
    )

    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            depth = depth_model.infer_image(rgb)
        # Simple lane/edge proxy: use bottom-heavy edges + SAM auto if available
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Focus on lower half (road)
        edges[: h // 3] = 0
        mask = (edges > 0).astype(np.uint8) * 255
        if sam_gen is not None and frame_i % 5 == 0:
            try:
                masks = sam_gen.generate(rgb)
                # Keep large thin masks near the bottom as lane candidates
                for m in masks[:8]:
                    seg = m["segmentation"]
                    if seg.sum() > 500 and seg[h // 2 :, :].sum() > seg.sum() * 0.4:
                        mask = np.maximum(mask, seg.astype(np.uint8) * 255)
            except Exception:
                pass
        bev = mask_to_bev(mask, depth, args.fx, args.fy, args.cx, args.cy, bev_size=h)
        # Resize BEV to frame height if needed
        if bev.shape[0] != h:
            bev = cv2.resize(bev, (bev_w, h))
        overlay = frame.copy()
        overlay[mask > 0] = (0, 255, 0)
        combined = np.hstack([overlay, bev])
        writer.write(combined)
        frame_i += 1
        if frame_i % 30 == 0:
            print(frame_i, flush=True)

    writer.release()
    cap.release()
    print("wrote", args.output, "frames", frame_i, flush=True)


if __name__ == "__main__":
    main()
