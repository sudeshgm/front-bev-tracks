#!/usr/bin/env python3
"""Lift YOLO-World boxes and YOLOPv2 lanes with UniDepth and this rig's calibration.

Qwen-Drive's 3D head was trained on a different camera ring. This pass does not
use it. Each camera is a normal image: open-vocabulary boxes, a lane mask, and
metric depth. The Audi intrinsics and extrinsics turn pixels into ego metres,
and the existing tracker keeps an id when a car moves from the side camera
into the front.
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

from qdtrack.geometry import CAM_SLOTS, pixel_to_ego  # noqa: E402
from qdtrack.tracks import DET_CLASS_NAMES, track_lanes, track_objects  # noqa: E402

PROMPTS = (
    ("car", "vehicle", (4.5, 1.8, 1.5)),
    ("bus", "vehicle", (12.0, 2.6, 3.0)),
    ("truck", "vehicle", (7.0, 2.5, 2.8)),
    ("person", "pedestrian", (0.6, 0.6, 1.7)),
    ("bicycle", "bicycle", (1.8, 0.6, 1.2)),
    ("motorcycle", "bicycle", (2.0, 0.7, 1.4)),
    ("traffic cone", "traffic_cone", (0.4, 0.4, 0.7)),
)
NAME_TO_LABEL = {name: DET_CLASS_NAMES.index(name) for name in DET_CLASS_NAMES}
LANE_CAMERAS = {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}
WIDTH = 640


def resize_keep(image: np.ndarray, intrinsic: np.ndarray):
    height, width = image.shape[:2]
    scale = WIDTH / width
    resized = cv2.resize(image, (WIDTH, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    scaled = np.array(intrinsic, dtype=np.float64, copy=True)
    scaled[0] *= scale
    scaled[1] *= scale
    return resized, scaled


def yolop_canvas(image: np.ndarray):
    """Paste the frame where YOLOPv2 expects a 1280x720 picture.

    The traced head was exported for that letterbox: content is 640x360,
    starting 140 rows down a 640x640 canvas. The returned mask is aligned
    with ``image``, not with the canvas.
    """
    height, width = image.shape[:2]
    content = cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    canvas[140 : 140 + 360, 0:640] = content
    return canvas, height, width


def head_mask(raw: torch.Tensor, drive: bool) -> np.ndarray:
    """Official YOLOPv2 decode: crop the 16:9 band, double it, then threshold."""
    if raw.ndim == 4 and raw.shape[-2] >= 372:
        raw = raw[:, :, 12:372, :]
    if drive and raw.ndim == 4 and raw.shape[1] > 1:
        raw = raw.argmax(dim=1, keepdim=True).float()
        up = F.interpolate(raw, scale_factor=2, mode="nearest")
        return up.squeeze().detach().cpu().numpy() == 1
    if raw.ndim == 4 and raw.shape[1] > 1:
        raw = raw[:, 1:2]
    up = F.interpolate(raw.float(), scale_factor=2, mode="bilinear", align_corners=False)
    if float(up.detach().min()) < -0.05 or float(up.detach().max()) > 1.5:
        up = torch.sigmoid(up)
    return up.squeeze().detach().cpu().numpy() > 0.5


def components(mask: np.ndarray, minimum: int) -> list[np.ndarray]:
    binary = (mask > 0.5).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary)
    pieces = []
    for index in range(1, count):
        ys, xs = np.where(labels == index)
        if len(xs) < minimum:
            continue
        order = np.argsort(ys)
        step = max(1, len(xs) // 40)
        picked = order[::step]
        pieces.append(np.stack([xs[picked], ys[picked]], axis=1).astype(np.float64))
    return pieces


def sample_depth(depth: np.ndarray, u: float, v: float) -> float:
    height, width = depth.shape
    x = int(round(u))
    y = int(round(v))
    if not (0 <= x < width and 0 <= y < height):
        return -1.0
    patch = depth[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
    valid = patch[np.isfinite(patch) & (patch > 0.3) & (patch < 80)]
    if len(valid) == 0:
        return -1.0
    return float(np.median(valid))


def lift_points(pixels, depth, intrinsic, rotation, translation) -> np.ndarray:
    lifted = []
    for u, v in pixels:
        z = sample_depth(depth, u, v)
        ego = pixel_to_ego(u, v, z, intrinsic, rotation, translation)
        if ego is None:
            continue
        if not (-2.0 < ego[2] < 4.0 and -20.0 < ego[1] < 40.0 and -30.0 < ego[0] < 80.0):
            continue
        lifted.append(ego[:2])
    if len(lifted) < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(lifted, dtype=np.float64)


def nms_centers(boxes: list[list[float]], scores: list[float], labels: list[int], radius: float = 2.2):
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    for index in order:
        center = np.array(boxes[index][:2])
        if any(np.linalg.norm(center - np.array(boxes[other][:2])) < radius for other in kept):
            continue
        kept.append(index)
    return [boxes[i] for i in kept], [scores[i] for i in kept], [labels[i] for i in kept]


def load_depth(device: str):
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
    model = model.to(device).eval()
    return model, Pinhole


def load_lanes(device: str, weights: Path):
    if not weights.exists():
        import urllib.request

        weights.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt"
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, weights)
    model = torch.jit.load(str(weights), map_location=device)
    model = model.to(device).eval()
    return model


def depth_of(model, pinhole_cls, image: np.ndarray, intrinsic: np.ndarray, device: str) -> np.ndarray:
    rgb = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1)
    camera = pinhole_cls(K=torch.from_numpy(intrinsic.astype(np.float32)))
    with torch.no_grad():
        prediction = model.infer(rgb, camera)
    depth = prediction["depth"]
    depth = depth.squeeze().detach().float().cpu().numpy()
    if depth.shape != image.shape[:2]:
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth


def lane_polylines(model, image: np.ndarray, depth, intrinsic, rotation, translation, device: str, report: dict):
    canvas, image_h, image_w = yolop_canvas(image)
    tensor = torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))).float().div(255.0)
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        _pred, drive, lanes = model(tensor)
    if not report["seen"]:
        report["seen"] = True
        print(f"yolopv2 drive {tuple(drive.shape)} lanes {tuple(lanes.shape)}", flush=True)
    lane_mask = head_mask(lanes, drive=False)
    drive_mask = head_mask(drive, drive=True)
    report["lane_px"] += int(lane_mask.sum())
    report["drive_px"] += int(drive_mask.sum())
    lines = []
    for piece in components(lane_mask.astype(np.float32), minimum=40):
        pixels = piece.copy()
        pixels[:, 0] *= image_w / lane_mask.shape[1]
        pixels[:, 1] *= image_h / lane_mask.shape[0]
        lifted = lift_points(pixels, depth, intrinsic, rotation, translation)
        if len(lifted) >= 2:
            lines.append(lifted)
    edges = []
    # The drivable-area rim is the road edge. Ignore the sky half of the mask.
    rim = drive_mask.copy()
    rim[: int(rim.shape[0] * 0.45)] = False
    for piece in components(rim.astype(np.float32), minimum=80):
        pixels = piece.copy()
        pixels[:, 0] *= image_w / rim.shape[1]
        pixels[:, 1] *= image_h / rim.shape[0]
        by_row: dict[int, list[float]] = {}
        for u, v in pixels:
            by_row.setdefault(int(v), []).append(float(u))
        left_rim, right_rim = [], []
        for v in sorted(by_row):
            cols = by_row[v]
            left_rim.append((min(cols), v))
            right_rim.append((max(cols), v))
        for rim_pts in (left_rim, right_rim):
            step = max(1, len(rim_pts) // 30)
            lifted = lift_points(rim_pts[::step], depth, intrinsic, rotation, translation)
            if len(lifted) >= 2:
                edges.append(lifted)
    return lines, edges


def detect(detector, image: np.ndarray, depth, intrinsic, rotation, translation):
    result = detector.predict(image, conf=0.25, verbose=False)[0]
    boxes, scores, labels = [], [], []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes, scores, labels
    names = result.names
    for xyxy, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
        prompt = names[int(cls)]
        spec = next((item for item in PROMPTS if item[0] == prompt), None)
        if spec is None:
            continue
        _prompt, class_name, (length, width, height) = spec
        x0, y0, x1, y1 = xyxy
        u = (x0 + x1) / 2
        v = y1 - 0.15 * (y1 - y0)
        z = sample_depth(depth, u, v)
        ego = pixel_to_ego(u, v, z, intrinsic, rotation, translation)
        if ego is None or not (-5 < ego[0] < 80 and -25 < ego[1] < 40):
            continue
        # The bonnet sits at about 4 m straight ahead in the front camera.
        if ego[0] < 7 and abs(ego[1]) < 1.8:
            continue
        bottom = float(ego[2])
        boxes.append([float(ego[0]), float(ego[1]), bottom, length, width, height, 0.0, 0.0, 0.0])
        scores.append(float(score))
        labels.append(NAME_TO_LABEL[class_name])
    return nms_centers(boxes, scores, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score", type=float, default=0.25)
    parser.add_argument("--weights", type=Path, default=Path("weights/yolopv2.pt"))
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    from ultralytics import YOLOWorld

    detector = YOLOWorld("yolov8s-worldv2.pt")
    detector.set_classes([item[0] for item in PROMPTS])
    depth_model, pinhole_cls = load_depth(device)
    lanes_model = load_lanes(device, args.weights)
    print("models ready", flush=True)

    folders = sorted(path for path in args.frames.iterdir() if (path / "frame.json").exists())
    if not folders:
        raise SystemExit(f"no frames in {args.frames}")
    packed = []
    report = {"seen": False, "lane_px": 0, "drive_px": 0}
    for folder in folders:
        meta = json.loads((folder / "frame.json").read_text())
        calib = np.load(folder / "calib.npz")
        frame_boxes: list[list[float]] = []
        frame_scores: list[float] = []
        frame_labels: list[int] = []
        maps = {"road_line": [], "road_edge": []}
        for index, (qwen, _key, _tag, _view) in enumerate(CAM_SLOTS):
            image = cv2.imread(str(folder / "images" / f"{qwen}.jpg"))
            if image is None:
                raise SystemExit(f"missing {qwen} in {folder}")
            image, intrinsic = resize_keep(image, calib["cam_intrinsic"][index])
            rotation = calib["sensor2lidar_rotation"][index].astype(np.float64)
            translation = calib["sensor2lidar_translation"][index].astype(np.float64)
            depth = depth_of(depth_model, pinhole_cls, image, intrinsic, device)
            boxes, scores, labels = detect(detector, image, depth, intrinsic, rotation, translation)
            frame_boxes.extend(boxes)
            frame_scores.extend(scores)
            frame_labels.extend(labels)
            if qwen in LANE_CAMERAS:
                lines, edges = lane_polylines(
                    lanes_model, image, depth, intrinsic, rotation, translation, device, report
                )
                maps["road_line"].extend(lines)
                maps["road_edge"].extend(edges)
        if frame_boxes:
            frame_boxes, frame_scores, frame_labels = nms_centers(frame_boxes, frame_scores, frame_labels)
        packed.append(
            {
                "token": meta["token"],
                "t": meta["pose"]["t"],
                "pose": meta["pose"],
                "boxes": np.asarray(frame_boxes, dtype=np.float64).reshape(-1, 9),
                "scores": np.asarray(frame_scores, dtype=np.float64),
                "labels": np.asarray(frame_labels, dtype=np.int64),
                "maps": maps,
            }
        )
        print(
            f"{meta['token']}: {len(frame_scores)} boxes, "
            f"{len(maps['road_line'])} lines, {len(maps['road_edge'])} edges",
            flush=True,
        )
    print(f"lane pixels {report['lane_px']}  drive pixels {report['drive_px']}", flush=True)

    objects = track_objects(packed, score_min=args.score, max_gap=10)
    lanes = track_lanes(packed)
    args.output.mkdir(parents=True, exist_ok=True)
    document = {
        "model": "yolov8s-worldv2 + unidepth-v2-vits14 + yolopv2",
        "perception": "lifted with A2D2 calibration",
        "dtype": "float32",
        "attn": "n/a",
        "score_min": args.score,
        "classes": list(DET_CLASS_NAMES),
        "map_classes": ["road_line", "road_edge"],
        "object_tracks": objects,
        "lane_tracks": lanes,
        "frames": [
            {
                "token": frame["token"],
                "detections": [
                    {
                        "label": int(label),
                        "name": DET_CLASS_NAMES[int(label)],
                        "score": round(float(score), 4),
                        "box": [round(float(value), 4) for value in box],
                    }
                    for box, score, label in zip(frame["boxes"], frame["scores"], frame["labels"])
                    if float(score) >= args.score
                ],
            }
            for frame in packed
        ],
    }
    dest = args.output / "eval.json"
    dest.write_text(json.dumps(document))
    summary = args.output / "summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Lift stack on A2D2",
                "",
                "- YOLO-World boxes, YOLOPv2 lanes, UniDepth V2 metric depth",
                "- lifted with this clip's intrinsics and extrinsics",
                f"- frames: {len(packed)}",
                f"- object tracks: {len(objects)}",
                f"- lane tracks: {len(lanes)}",
                f"- score floor: {args.score}",
                "",
            ]
        )
    )
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
