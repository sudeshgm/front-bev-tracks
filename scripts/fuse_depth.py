#!/usr/bin/env python3
"""YOLOPv2 lanes and boxes, kept only where UniDepth agrees with the ground plane.

The 2D model is the one that already looked right on this clip. UniDepth supplies
a metric depth, and the Audi calibration supplies the flat-road depth of the
same pixel. A point is kept when those two ranges agree within 30 percent and
the depth point sits near the ground. Front camera only. The existing tracker
then stitches the kept points across the clip.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdtrack.geometry import CAM_SLOTS, pixel_to_ego  # noqa: E402
from qdtrack.tracks import DET_CLASS_NAMES, track_lanes, track_objects  # noqa: E402

AGREE = 0.30
VEHICLE = DET_CLASS_NAMES.index("vehicle")


def resize_to(image: np.ndarray, intrinsic: np.ndarray, width: int = 960):
    height, src_w = image.shape[:2]
    scale = width / src_w
    resized = cv2.resize(image, (width, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    scaled = np.array(intrinsic, dtype=np.float64, copy=True)
    scaled[0] *= scale
    scaled[1] *= scale
    return resized, scaled


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def yolop_forward(model, image: np.ndarray, device: str):
    """Return the lane mask and boxes in the coordinates of ``image``.

    The traced head emits a 640x640 map aligned with a letterboxed canvas,
    not the 12:372 crop used by the official 1280x720 demo.
    """
    height, width = image.shape[:2]
    content = cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[140:500] = content
    tensor = torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)))
    tensor = tensor.float().div(255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        (preds, anchors), _drive, lanes = model(tensor)
    prob = lanes[0, 0]
    if float(prob.detach().max()) > 1.5 or float(prob.detach().min()) < -0.05:
        prob = torch.sigmoid(prob)
    band = (prob[140:500] > 0.5).detach().float().cpu().numpy()
    mask = cv2.resize(band, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5

    parts = []
    for index, stride in enumerate((8, 16, 32)):
        batch, _channels, ny, nx = preds[index].shape
        pred = preds[index].reshape(batch, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).sigmoid().clone()
        gy, gx = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij")
        grid = torch.stack((gx, gy), 2).view(1, 1, ny, nx, 2)
        pred[..., 0:2] = (pred[..., 0:2] * 2 - 0.5 + grid) * stride
        pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * anchors[index].to(device=pred.device, dtype=pred.dtype)
        parts.append(pred.reshape(batch, -1, 85))
    pred = torch.cat(parts, 1)[0]
    pred = pred[pred[:, 4] > 0.25]
    boxes = []
    if len(pred):
        conf, _cls = pred[:, 5:].max(1)
        score = conf * pred[:, 4]
        keep = score > 0.35
        pred, score = pred[keep], score[keep]
        if len(pred):
            xy = pred[:, :4]
            xyxy = torch.stack(
                [xy[:, 0] - xy[:, 2] / 2, xy[:, 1] - xy[:, 3] / 2, xy[:, 0] + xy[:, 2] / 2, xy[:, 1] + xy[:, 3] / 2],
                1,
            )
            order = torch.argsort(score, descending=True)
            kept = []
            for i in order.tolist():
                box = xyxy[i].tolist()
                if any(iou(box, xyxy[j].tolist()) > 0.45 for j in kept):
                    continue
                kept.append(i)
            for i in kept:
                x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                boxes.append(
                    (
                        x1 / 640 * width,
                        (y1 - 140) / 360 * height,
                        x2 / 640 * width,
                        (y2 - 140) / 360 * height,
                        float(score[i]),
                    )
                )
    return mask, boxes


def load_lanes(device: str, weights: Path):
    if not weights.exists():
        import urllib.request

        weights.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, weights)
    model = torch.jit.load(str(weights), map_location=device).to(device).eval()
    return model


def load_depth(device: str):
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14").to(device).eval()
    return model, Pinhole


def depth_of(model, pinhole_cls, image: np.ndarray, intrinsic: np.ndarray, device: str) -> np.ndarray:
    rgb = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1)
    camera = pinhole_cls(K=torch.from_numpy(intrinsic.astype(np.float32)))
    with torch.no_grad():
        prediction = model.infer(rgb, camera)
    depth = prediction["depth"].squeeze().detach().float().cpu().numpy()
    if depth.shape != image.shape[:2]:
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth


def sample_depth(depth: np.ndarray, u: float, v: float) -> float:
    height, width = depth.shape
    x, y = int(round(u)), int(round(v))
    if not (0 <= x < width and 0 <= y < height):
        return -1.0
    patch = depth[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
    valid = patch[np.isfinite(patch) & (patch > 0.4) & (patch < 80)]
    if len(valid) == 0:
        return -1.0
    return float(np.median(valid))


def ground_depth(u: float, v: float, intrinsic, rotation, translation):
    """Range along the ray at which it meets the ego ground plane z = 0."""
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx == 0 or fy == 0:
        return None
    direction = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    denom = float(np.asarray(rotation, dtype=np.float64)[2] @ direction)
    if abs(denom) < 1e-8:
        return None
    depth = -float(np.asarray(translation, dtype=np.float64)[2]) / denom
    if not (0.4 < depth < 60):
        return None
    return depth


def agreed(u, v, depth_map, intrinsic, rotation, translation):
    plane = ground_depth(u, v, intrinsic, rotation, translation)
    measured = sample_depth(depth_map, u, v)
    if plane is None or measured < 0:
        return None
    if abs(measured - plane) / plane > AGREE:
        return None
    ego = pixel_to_ego(u, v, measured, intrinsic, rotation, translation)
    if ego is None or abs(float(ego[2])) > 1.0:
        return None
    if not (2.0 < float(ego[0]) < 40.0 and -15.0 < float(ego[1]) < 15.0):
        return None
    return ego


def centerlines(mask: np.ndarray):
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    lines = []
    for index in range(1, count):
        ys, xs = np.where(labels == index)
        if len(xs) < 80:
            continue
        rows: dict[int, list[int]] = {}
        for x, y in zip(xs, ys):
            rows.setdefault(int(y), []).append(int(x))
        pts = [(float(np.mean(rows[y])), float(y)) for y in sorted(rows)]
        step = max(1, len(pts) // 40)
        lines.append(pts[::step])
    return lines


def fuse_line(pixels, depth_map, intrinsic, rotation, translation, stats: dict) -> np.ndarray:
    kept = []
    for u, v in pixels:
        stats["lane_seen"] += 1
        ego = agreed(u, v, depth_map, intrinsic, rotation, translation)
        if ego is None:
            continue
        stats["lane_kept"] += 1
        kept.append(ego[:2])
    if len(kept) < 4:
        return np.zeros((0, 2), dtype=np.float64)
    poly = np.asarray(kept, dtype=np.float64)
    if float(np.linalg.norm(poly[-1] - poly[0])) < 4.0:
        return np.zeros((0, 2), dtype=np.float64)
    return poly


def fuse_box(x1, y1, x2, y2, score, depth_map, intrinsic, rotation, translation, stats: dict):
    stats["box_seen"] += 1
    bottom = max(y1, y2)
    left_x, right_x = min(x1, x2), max(x1, x2)
    center = agreed((left_x + right_x) / 2, bottom, depth_map, intrinsic, rotation, translation)
    if center is None:
        return None
    left = agreed(left_x, bottom, depth_map, intrinsic, rotation, translation)
    right = agreed(right_x, bottom, depth_map, intrinsic, rotation, translation)
    if left is not None and right is not None:
        edge = float(np.linalg.norm(right[:2] - left[:2]))
    else:
        edge = 0.0
    mx, my = float(center[0]), float(center[1])
    if edge >= 3.4:
        length = min(edge, 14.0)
        width = 2.5
        height = 3.0
        dx, dy = float(right[0] - left[0]), float(right[1] - left[1])
        yaw = math.atan2(dy, dx)
        nx, ny = -dy / max(edge, 1e-6), dx / max(edge, 1e-6)
        if ny * my < 0:
            nx, ny = -nx, -ny
        cx = (float(left[0]) + float(right[0])) / 2 + nx * width / 2
        cy = (float(left[1]) + float(right[1])) / 2 + ny * width / 2
    else:
        width = 1.9
        length = 4.6
        height = 1.6
        yaw = 0.0
        cx, cy = mx + length / 2, my
    if cx < 6.0 and abs(cy) < 1.8:
        return None
    if not (2.0 < cx < 45.0 and -14.0 < cy < 14.0):
        return None
    stats["box_kept"] += 1
    return [cx, cy, height / 2, width, length, height, yaw, 0.0, 0.0], score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("data/frames"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score", type=float, default=0.35)
    parser.add_argument("--weights", type=Path, default=Path("weights/yolopv2.pt"))
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    depth_model, pinhole_cls = load_depth(device)
    lanes_model = load_lanes(device, args.weights)
    print("models ready", flush=True)
    front = CAM_SLOTS.index(next(slot for slot in CAM_SLOTS if slot[0] == "CAM_FRONT"))

    folders = sorted(path for path in args.frames.iterdir() if (path / "frame.json").exists())
    if not folders:
        raise SystemExit(f"no frames in {args.frames}")
    packed = []
    stats = {"lane_seen": 0, "lane_kept": 0, "box_seen": 0, "box_kept": 0}
    for folder in folders:
        meta = json.loads((folder / "frame.json").read_text())
        calib = np.load(folder / "calib.npz")
        image = cv2.imread(str(folder / "images" / "CAM_FRONT.jpg"))
        if image is None:
            raise SystemExit(f"missing CAM_FRONT in {folder}")
        image, intrinsic = resize_to(image, calib["cam_intrinsic"][front])
        rotation = calib["sensor2lidar_rotation"][front].astype(np.float64)
        translation = calib["sensor2lidar_translation"][front].astype(np.float64)
        depth = depth_of(depth_model, pinhole_cls, image, intrinsic, device)
        mask, boxes = yolop_forward(lanes_model, image, device)
        lines = []
        for pixels in centerlines(mask):
            poly = fuse_line(pixels, depth, intrinsic, rotation, translation, stats)
            if len(poly):
                lines.append(poly)
        frame_boxes, frame_scores, frame_labels = [], [], []
        for x1, y1, x2, y2, score in boxes:
            fused = fuse_box(x1, y1, x2, y2, score, depth, intrinsic, rotation, translation, stats)
            if fused is None:
                continue
            box, kept_score = fused
            frame_boxes.append(box)
            frame_scores.append(kept_score)
            frame_labels.append(VEHICLE)
        packed.append(
            {
                "token": meta["token"],
                "t": meta["pose"]["t"],
                "pose": meta["pose"],
                "boxes": np.asarray(frame_boxes, dtype=np.float64).reshape(-1, 9),
                "scores": np.asarray(frame_scores, dtype=np.float64),
                "labels": np.asarray(frame_labels, dtype=np.int64),
                "maps": {"road_line": lines, "road_edge": []},
            }
        )
        print(
            f"{meta['token']}: {len(frame_scores)} boxes, {len(lines)} lines",
            flush=True,
        )

    objects = track_objects(packed, score_min=args.score, max_gap=8)
    lanes = track_lanes(packed, max_gap=8, collapse=False)
    args.output.mkdir(parents=True, exist_ok=True)
    document = {
        "model": "yolopv2 + unidepth-v2-vits14",
        "perception": "front camera, kept where UniDepth agrees with the ground plane within 30%",
        "dtype": "float32",
        "attn": "n/a",
        "score_min": args.score,
        "classes": list(DET_CLASS_NAMES),
        "map_classes": ["road_line"],
        "object_tracks": objects,
        "lane_tracks": lanes,
        "frames": [
            {
                "token": frame["token"],
                "detections": [
                    {
                        "label": int(label),
                        "name": "vehicle",
                        "score": round(float(score), 4),
                        "box": [round(float(value), 4) for value in box],
                    }
                    for box, score, label in zip(frame["boxes"], frame["scores"], frame["labels"])
                ],
            }
            for frame in packed
        ],
    }
    dest = args.output / "eval.json"
    dest.write_text(json.dumps(document))
    lane_rate = stats["lane_kept"] / max(stats["lane_seen"], 1)
    box_rate = stats["box_kept"] / max(stats["box_seen"], 1)
    summary = args.output / "summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Depth agreement on A2D2",
                "",
                "- YOLOPv2 lanes and boxes, front camera only",
                "- UniDepth V2 metric depth, kept when it matches the ground plane within 30%",
                "- then tracked in the ego frame",
                f"- frames: {len(packed)}",
                f"- object tracks: {len(objects)}",
                f"- lane tracks: {len(lanes)}",
                f"- lane points kept: {stats['lane_kept']} / {stats['lane_seen']} ({lane_rate:.0%})",
                f"- boxes kept: {stats['box_kept']} / {stats['box_seen']} ({box_rate:.0%})",
                "",
            ]
        )
    )
    print(
        f"kept lanes {stats['lane_kept']}/{stats['lane_seen']}  boxes {stats['box_kept']}/{stats['box_seen']}",
        flush=True,
    )
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
