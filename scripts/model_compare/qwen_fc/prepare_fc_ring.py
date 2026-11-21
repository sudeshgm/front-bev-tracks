"""
Qwen-Drive perception, front-camera only (Apache-2.0).

Uses only the perception head + VLM feature taps.
Other cameras in the 6-cam ring are black images.
KITTI front camera is placed as CAM_FRONT; remaining views are black.

Requires the official Qwen-Drive-1.0 repo and weights on the GPU machine.
This script prepares the fake ring and calls the official perception path.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

# Standard 6-cam nuScenes-style names used by the demo
CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

def make_black(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)

def prepare_frame(fc_path, out_dir, size=(896, 512)):
    """Create a frame directory with FC real + 5 black images."""
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    fc = cv2.imread(str(fc_path))
    if fc is None:
        raise FileNotFoundError(fc_path)
    fc = cv2.resize(fc, size)

    # Approximate KITTI-like intrinsics scaled to the resized image
    # (GPU Bot can replace with real KITTI calib if available)
    fx = fy = 0.7 * size[0]
    cx, cy = size[0] / 2, size[1] / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    images = {}
    for i, cam in enumerate(CAM_ORDER):
        if cam == "CAM_FRONT":
            img = fc
        else:
            img = make_black(*size[::-1])
        p = img_dir / f"{cam}.jpg"
        cv2.imwrite(str(p), img)
        images[cam] = str(p)

    # Minimal calib (identity-ish extrinsics; GPU Bot should refine with real KITTI if present)
    cam_intrinsic = np.stack([K] * 6)
    sensor2lidar_rotation = np.stack([np.eye(3)] * 6)
    sensor2lidar_translation = np.zeros((6, 3), dtype=np.float32)
    # simple azimuth offsets so the ring is not all at the same pose
    for i, yaw in enumerate([0, -np.pi/3, -2*np.pi/3, np.pi, 2*np.pi/3, np.pi/3]):
        c, s = np.cos(yaw), np.sin(yaw)
        sensor2lidar_rotation[i] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    lidar2ego = np.eye(4, dtype=np.float32)

    np.savez(
        out / "calib.npz",
        cam_intrinsic=cam_intrinsic,
        sensor2lidar_rotation=sensor2lidar_rotation,
        sensor2lidar_translation=sensor2lidar_translation,
        lidar2ego=lidar2ego,
    )

    # empty gt / lidar placeholders so the official loader does not crash
    np.savez(out / "gt.npz", boxes=np.zeros((0, 7)), labels=np.zeros((0,), dtype=np.int64))
    np.save(out / "lidar.npy", np.zeros((0, 4), dtype=np.float32))

    frame = {
        "token": Path(fc_path).stem,
        "dataset_type": "nuscenes",
        "cameras": CAM_ORDER,
        "images": {c: f"images/{c}.jpg" for c in CAM_ORDER},
    }
    (out / "frame.json").write_text(json.dumps(frame, indent=2))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fc-image", required=True, help="front-camera frame (jpg/png)")
    p.add_argument("--out-frame", required=True, help="output frame directory")
    p.add_argument("--size", nargs=2, type=int, default=[896, 512])
    args = p.parse_args()
    prepare_frame(args.fc_image, args.out_frame, tuple(args.size))
    print("Prepared FC+black ring at", args.out_frame)
    print("Next: run official scripts/run_perception.py --frames", args.out_frame)

if __name__ == "__main__":
    main()
