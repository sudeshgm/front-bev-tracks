"""
Qwen-Drive perception, front-camera only (Apache-2.0).

Uses only the perception head + VLM feature taps.
Other cameras in the 6-cam ring are black images.
KITTI front camera is placed as CAM_FRONT; remaining views are black.

Requires the official Qwen-Drive-1.0 repo and weights on the GPU machine.
This script prepares the fake ring for the official perception path.

GPU Bot fix (separate commit on qwen-fc-results), why:
  * The official loader (qwen_drive_perception/dataset.py PerceptionFrame) reads
    frame.json keys "cam_order" and "content" (view tag / image pairs + prompt).
    The first version wrote "cameras"/"images" -> KeyError.
  * Calibration used identity camera->lidar rotations. A camera frame is
    x-right, y-down, z-forward, so identity points every optical axis at the sky
    and the BEV lifting is meaningless. Now: the 5 black cameras keep the real
    nuScenes calibration of the official nuScenes demo frame (--template-frame),
    lidar2ego is that frame's real LIDAR_TOP->ego, and CAM_FRONT gets a KITTI
    camera pose (optical axis = ego forward, 1.65 m above ground, ~1.7 m ahead
    of the rear axle) and the KITTI P2 intrinsics (2011_09_26 calib).
  * The prepared clips are the full KITTI 1242x375 frame squeezed to 640x480
    (verified: lane width vs horizon distance is off by ~2.5x). The frame is
    restored to 1242x375 and letterboxed with black to the 896x512 aspect, so
    the model's own resize to 896x512 is uniform; cy is shifted by the pad.
  * gt.npz carries empty occ/map/box arrays in the official shapes.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CAM_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]
VIEW_TAGS = {
    "CAM_FRONT": "<FRONT VIEW>",
    "CAM_FRONT_RIGHT": "<FRONT RIGHT VIEW>",
    "CAM_BACK_RIGHT": "<BACK RIGHT VIEW>",
    "CAM_BACK": "<BACK VIEW>",
    "CAM_BACK_LEFT": "<BACK LEFT VIEW>",
    "CAM_FRONT_LEFT": "<FRONT LEFT VIEW>",
}
PROMPT = "Analyze the scene."  # same instruction as the official nuScenes demo frames

KITTI_SIZE = (1242, 375)                       # (w, h) of KITTI raw image_02
KITTI_P2 = (721.5377, 721.5377, 609.5593, 172.854)  # fx, fy, cx, cy (2011_09_26)
KITTI_CAM_HEIGHT = 1.65                         # m above ground
KITTI_CAM_FORWARD = 1.70                        # m ahead of ego origin (rear axle), approx
MODEL_ASPECT = 896 / 512
# camera (x right, y down, z forward) -> ego-like axes (x forward, y left, z up)
CAM2EGO_ROT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


def kitti_front(fc_bgr, stretched=True):
    """Return (image, K, pad_top). Undo the 640x480 squeeze and letterbox to 896:512."""
    img = cv2.resize(fc_bgr, KITTI_SIZE, interpolation=cv2.INTER_CUBIC) if stretched else fc_bgr
    w, h = img.shape[1], img.shape[0]
    target_h = int(round(w / MODEL_ASPECT))
    pad_top = max(0, (target_h - h) // 2)
    pad_bot = max(0, target_h - h - pad_top)
    img = cv2.copyMakeBorder(img, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    fx, fy, cx, cy = KITTI_P2
    K = np.array([[fx, 0, cx], [0, fy, cy + pad_top], [0, 0, 1]], dtype=np.float64)
    return img, K, pad_top


def prepare_frame(fc_path, out_dir, template_frame, stretched=True):
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    tpl = Path(template_frame)
    tcal = np.load(tpl / "calib.npz")
    tframe = json.loads((tpl / "frame.json").read_text())
    if tframe["cam_order"] != CAM_ORDER:
        raise SystemExit(f"template camera order {tframe['cam_order']} != {CAM_ORDER}")

    fc = cv2.imread(str(fc_path))
    if fc is None:
        raise FileNotFoundError(fc_path)
    front, K_front, pad_top = kitti_front(fc, stretched)

    cam_intrinsic = np.array(tcal["cam_intrinsic"], dtype=np.float64).copy()
    rot = np.array(tcal["sensor2lidar_rotation"], dtype=np.float64).copy()
    trans = np.array(tcal["sensor2lidar_translation"], dtype=np.float64).copy()
    lidar2ego = np.array(tcal["lidar2ego"], dtype=np.float64).copy()

    # CAM_FRONT pose: camera->ego is CAM2EGO_ROT at (forward, 0, height); express in the lidar frame.
    ego2lidar = np.linalg.inv(lidar2ego)
    rot[0] = ego2lidar[:3, :3] @ CAM2EGO_ROT
    trans[0] = ego2lidar[:3, :3] @ np.array([KITTI_CAM_FORWARD, 0.0, KITTI_CAM_HEIGHT]) + ego2lidar[:3, 3]
    cam_intrinsic[0] = K_front

    for i, cam in enumerate(CAM_ORDER):
        if cam == "CAM_FRONT":
            img = front
        else:
            from PIL import Image
            w, h = Image.open(tpl / "images" / f"{cam}.jpg").size  # keep the template image size
            img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{cam}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    np.savez(out / "calib.npz", cam_intrinsic=cam_intrinsic, sensor2lidar_rotation=rot,
             sensor2lidar_translation=trans, lidar2ego=lidar2ego)
    np.savez(out / "gt.npz", occ=np.zeros((200, 200, 16), np.int8), map=np.zeros((200, 400), np.int8),
             boxes=np.zeros((0, 9), np.float32), labels=np.zeros((0,), np.int64))
    np.save(out / "lidar.npy", np.zeros((0, 4), dtype=np.float32))

    content = []
    for cam in CAM_ORDER:
        content += [{"text": VIEW_TAGS[cam]}, {"image": cam}]
    content.append({"text": PROMPT})
    frame = {
        "dataset_type": "nuscenes",
        "cam_order": CAM_ORDER,
        "content": content,
        "source": {"fc_image": str(fc_path), "kitti_stretched_input": stretched, "pad_top": pad_top,
                   "kitti_P2": KITTI_P2, "template_frame": tpl.name},
    }
    (out / "frame.json").write_text(json.dumps(frame, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fc-image", required=True, help="front-camera frame (jpg/png)")
    p.add_argument("--out-frame", required=True, help="output frame directory")
    p.add_argument("--template-frame", required=True,
                   help="official nuScenes demo frame dir, e.g. Qwen-Drive-1.0/data/demo/perception/90162f90eceb4ada9e595bc1adb71b5f")
    p.add_argument("--not-stretched", action="store_true", help="input is already the native 1242x375 KITTI frame")
    args = p.parse_args()
    prepare_frame(args.fc_image, args.out_frame, args.template_frame, stretched=not args.not_stretched)
    print("Prepared FC+black ring at", args.out_frame)


if __name__ == "__main__":
    main()
