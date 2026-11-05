"""Front camera and bird's-eye overlay for one driving clip.

The clip is not stored in this repo. The caller supplies three files:

- a video whose frames are already the virtual camera image
- a motion CSV with one row per camera frame
- a camera JSON for the CylinderV01 view those frames are in

Run:

    python scripts/overlay.py \
        --video clip.mp4 \
        --motion clip_motion.csv \
        --camera clip_camera.json \
        --weights yolopv2.pt \
        --output overlay.mp4

If the file repeats every camera frame, pass --repeat 2. The motion CSV still
has one row per camera frame, and the output is one frame per row.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdtrack.tracks import fit_lane_curves, integrate_ego, track_objects, world_to_ego, extend_tracks

MODEL = 640
CONTENT_H = 360
PAD_Y = (MODEL - CONTENT_H) // 2


class CylinderCamera:
    """CylinderV01 virtual view. The image is already unrotated.

    Vehicle axes are x forward, y left, z up. The camera position is in that
    frame. A ground point at (x, y, 0) is moved into the camera by subtracting
    the camera position (the view rotation is identity), then

        u = cx - fx * atan2(y, x)
        v = cy - fy * z / hypot(x, y)

    with no pixel when x <= 0.
    """

    def __init__(self, fx, fy, cx, cy, position, width, height):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.position = np.asarray(position, dtype=np.float64)
        self.width = int(width)
        self.height = int(height)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(
            data["fx"],
            data["fy"],
            data["cx"],
            data["cy"],
            data["position_m"],
            data["width"],
            data["height"],
        )

    def lift(self, u, v):
        """Pixel to the ground point in the vehicle frame. None above the horizon."""
        if v <= self.cy + 1:
            return None
        phi = (self.cx - float(u)) / self.fx
        direction = np.array([np.cos(phi), np.sin(phi), (self.cy - float(v)) / self.fy])
        if direction[2] >= -1e-4:
            return None
        scale = -self.position[2] / direction[2]
        point = self.position + scale * direction
        if not (4.0 < point[0] < 45.0 and abs(point[1]) < 14.0):
            return None
        return point

    def project(self, x, y, z=0.0):
        cam = np.array([x, y, z], dtype=np.float64) - self.position
        if cam[0] <= 0.5:
            return None
        u = self.cx - self.fx * np.arctan2(cam[1], cam[0])
        v = self.cy - self.fy * cam[2] / np.hypot(cam[0], cam[1])
        return float(u), float(v)

    def project_many(self, xy):
        """Ground points (N, 2) in the vehicle frame to pixels. NaN when x <= 0.5."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        delta = xy - self.position[:2]
        out = np.full((len(xy), 2), np.nan)
        valid = delta[:, 0] > 0.5
        if not np.any(valid):
            return out
        hyp = np.hypot(delta[:, 0], delta[:, 1])
        out[valid, 0] = self.cx - self.fx * np.arctan2(delta[valid, 1], delta[valid, 0])
        out[valid, 1] = self.cy + self.fy * self.position[2] / np.maximum(hyp[valid], 1e-6)
        return out


def load_motion(path):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    samples = [
        {"t": float(row["t_s"]), "speed_mps": float(row["speed_mps"]), "yaw_rate": float(row["yaw_rate_rps"])}
        for row in rows
    ]
    if len(samples) < 2:
        raise SystemExit("the motion file needs at least two rows")
    steps = np.diff([sample["t"] for sample in samples])
    if np.any(steps <= 0):
        raise SystemExit("t_s must increase")
    return samples, integrate_ego(samples), float(np.median(steps))


def decode_boxes(pred, camera, conf=0.4):
    feats, anchors = pred
    strides = [8, 16, 32]
    candidates = []
    for index, feat in enumerate(feats):
        _, _, ny, nx = feat.shape
        scores = feat.view(1, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).sigmoid()[0]
        anchor = anchors[index].view(3, 1, 1, 2)
        gy, gx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
        grid = torch.stack((gx, gy), -1).float().view(1, ny, nx, 2)
        center = (scores[..., 0:2] * 2 - 0.5 + grid) * strides[index]
        size = (scores[..., 2:4] * 2) ** 2 * anchor
        score, _kind = scores[..., 5:].max(-1)
        score = score * scores[..., 4]
        keep = score > conf
        if keep.any():
            candidates.append(
                torch.cat(
                    [center[keep] - size[keep] / 2, center[keep] + size[keep] / 2, score[keep, None]],
                    1,
                )
            )
    if not candidates:
        return []
    boxes = torch.cat(candidates).detach()
    order = boxes[:, 4].argsort(descending=True)
    kept = []
    drawn = []
    for index in order.tolist():
        box = boxes[index]
        if any(_iou(box, boxes[other]) > 0.45 for other in kept):
            continue
        x1, y1, x2, y2, score = box.tolist()
        if (x2 - x1) * (y2 - y1) < 80:
            continue
        p1 = (x1 / MODEL * camera.width, (y1 - PAD_Y) / CONTENT_H * camera.height)
        p2 = (x2 / MODEL * camera.width, (y2 - PAD_Y) / CONTENT_H * camera.height)
        if p2[0] < 0 or p1[0] > camera.width or p2[1] < 0 or p1[1] > camera.height:
            continue
        kept.append(index)
        drawn.append((p1[0], p1[1], p2[0], p2[1], score))
    return drawn


def _iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    area += (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    area -= inter
    return inter / area if area > 0 else 0.0


def lane_polylines(mask, camera):
    """Centre of each lane-mask run, linked down the image. Pixels, not metres."""
    if mask is None:
        return []
    row_index = range(int(camera.cy) + 6, camera.height - 8, 3)
    rows = []
    for v in row_index:
        cols = np.flatnonzero(mask[v])
        if len(cols) == 0:
            rows.append([])
            continue
        splits = np.where(np.diff(cols) > 6)[0] + 1
        points = []
        for run in np.split(cols, splits):
            if not (2 <= len(run) <= 40):
                continue
            points.append((float(run.mean()), float(v)))
        rows.append(points)
    used = [set() for _ in rows]
    lines = []
    for r, points in enumerate(rows):
        for i, point in enumerate(points):
            if i in used[r]:
                continue
            chain = [point]
            used[r].add(i)
            prev = point
            misses = 0
            for n in range(r + 1, min(r + 120, len(rows))):
                if not rows[n]:
                    misses += 1
                    if misses > 2:
                        break
                    continue
                misses = 0
                best, best_d = None, 48.0
                for j, cand in enumerate(rows[n]):
                    if j in used[n]:
                        continue
                    dist = abs(cand[0] - prev[0])
                    if dist < best_d:
                        best, best_d = j, dist
                if best is None:
                    break
                used[n].add(best)
                prev = rows[n][best]
                chain.append(prev)
            chain = np.asarray(chain, dtype=np.float64)
            if len(chain) >= 6 and np.linalg.norm(np.diff(chain, axis=0), axis=1).sum() >= 30.0:
                lines.append(chain[::2])
    return lines


def ground_contact(camera, x1, y1, x2, y2):
    """Ground point under a box. A box cut off by the frame uses the last visible row."""
    v = min(float(y2), camera.height - 2.0)
    return camera.lift((float(x1) + float(x2)) / 2.0, v)


def kept_boxes(rows):
    """Drop a box whose centre lies inside a larger one. It is the same car seen twice."""
    ordered = sorted(rows, key=lambda row: -((row[2] - row[0]) * (row[3] - row[1])))
    kept = []
    for row in ordered:
        cx = (row[0] + row[2]) / 2.0
        cy = (row[1] + row[3]) / 2.0
        if any(other[0] <= cx <= other[2] and other[1] <= cy <= other[3] for other in kept):
            continue
        kept.append(row)
    return kept


def kalman_positions(track, poses, frame_dt):
    """Constant-velocity Kalman, run both ways.

    The box bottom is a noisy range, not a speed. The filter keeps the speed
    the positions support.
    """
    measurements = []
    headings = []
    steps = []
    previous = None
    for sample in track["samples"]:
        index = int(sample["token"])
        measurements.append(sample["world"])
        headings.append(poses[index]["yaw"])
        steps.append(frame_dt if previous is None else max(index - previous, 1) * frame_dt)
        previous = index
    measurements = np.asarray(measurements, dtype=np.float64)
    count = len(measurements)
    span = min(8, count - 1)
    velocity = (measurements[span] - measurements[0]) / max(sum(steps[1 : span + 1]), 1e-3)
    state = np.array([measurements[0, 0], measurements[0, 1], velocity[0], velocity[1]])
    covariance = np.diag([4.0, 4.0, 9.0, 9.0])
    identity = np.eye(4)
    states = np.zeros((count, 4))
    covariances = []
    motions = []
    noises = []
    for index in range(count):
        dt = steps[index]
        motion = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        cosine, sine = np.cos(headings[index]), np.sin(headings[index])
        # A car brakes along the road. It does not accelerate sideways. The
        # sideways motion of a close box is the bottom of the box sliding
        # across the car, and following it turns the heading.
        along_accel = np.array([0.5 * dt * dt * cosine, 0.5 * dt * dt * sine, dt * cosine, dt * sine])
        side_accel = np.array([-0.5 * dt * dt * sine, 0.5 * dt * dt * cosine, -dt * sine, dt * cosine])
        noise = 3.0 ** 2 * np.outer(along_accel, along_accel) + 0.3 ** 2 * np.outer(side_accel, side_accel)
        state = motion @ state
        covariance = motion @ covariance @ motion.T + noise
        distance = abs(float(track["samples"][index]["box"][0])) if track["samples"][index].get("box") else 20.0
        sigma_along = min(6.0, max(0.8, distance * distance * 5.0 / 590.0))
        sigma_side = 2.0
        axes = np.array(
            [[cosine * sigma_along, -sine * sigma_side], [sine * sigma_along, cosine * sigma_side]]
        )
        measure = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        innovation = measurements[index] - measure @ state
        gain_scale = measure @ covariance @ measure.T + axes @ axes.T
        gain = covariance @ measure.T @ np.linalg.inv(gain_scale)
        state = state + gain @ innovation
        covariance = (identity - gain @ measure) @ covariance
        states[index] = state
        covariances.append(covariance.copy())
        motions.append(motion)
        noises.append(noise)
    for index in range(count - 2, -1, -1):
        motion = motions[index + 1]
        predicted = motion @ covariances[index] @ motion.T + noises[index + 1]
        carry = covariances[index] @ motion.T @ np.linalg.inv(predicted)
        states[index] = states[index] + carry @ (states[index + 1] - motion @ states[index])
        covariances[index] = covariances[index] + carry @ (covariances[index + 1] - predicted) @ carry.T
    return states[:, :2]


def smooth_world(tracks, poses, frame_dt):
    """Hold a car that is still. Do not decide that from how fast the range closes.

    The bottom of the box is the rear bumper from behind and a side corner
    once the car is abreast. If that sideways motion follows the viewing angle
    and the car has not moved along the road, keep the head-on position.
    A car whose fitted world speed is about zero is held the same way. An
    oncoming car closes faster than the ego vehicle drives, but the measured
    range is compressed, so a range test would freeze it. Its world speed is
    not zero, and that is what is kept.
    """
    for track in tracks:
        predicted = [sample for sample in track["samples"] if sample.get("predicted")]
        track["samples"] = [sample for sample in track["samples"] if not sample.get("predicted")]
        worlds = np.array([sample["world"] for sample in track["samples"]], dtype=np.float64)
        if len(worlds) < 3:
            track["samples"].extend(predicted)
            continue
        bearings = np.array(
            [abs(np.arctan2(sample["box"][1], max(sample["box"][0], 0.1))) for sample in track["samples"]]
        )
        order = np.argsort(bearings)[: max(3, len(bearings) // 5)]
        reference = np.median(worlds[order], axis=0)
        offset = worlds - reference
        yaw = float(np.mean([poses[int(track["samples"][i]["token"])]["yaw"] for i in order]))
        along = offset[:, 0] * np.cos(yaw) + offset[:, 1] * np.sin(yaw)
        lateral = -offset[:, 0] * np.sin(yaw) + offset[:, 1] * np.cos(yaw)
        correlation = 0.0 if lateral.std() < 1e-6 else float(np.corrcoef(bearings, lateral)[0, 1])
        passing = len(worlds) >= 8 and abs(correlation) > 0.8 and lateral.std() > 0.8 and along.std() < 8.0
        path = kalman_positions(track, poses, frame_dt)
        speed = 0.0
        if len(path) >= 2:
            step = np.linalg.norm(np.diff(path, axis=0), axis=1)
            elapsed = []
            for left, right in zip(track["samples"], track["samples"][1:]):
                elapsed.append(max(int(right["token"]) - int(left["token"]), 1) * frame_dt)
            speed = float(np.median(step / np.maximum(elapsed, 1e-3)))
        if passing:
            path = np.repeat(reference.reshape(1, 2), len(worlds), axis=0)
        elif speed < 0.8:
            path = np.repeat(np.median(path, axis=0).reshape(1, 2), len(worlds), axis=0)
        for sample, point in zip(track["samples"], path):
            sample["world"] = [float(point[0]), float(point[1])]
        track["samples"].extend(predicted)


def open_frames(video, frames, camera, repeat):
    """BGR frames in motion-row order, either from a video or from image files.

    Image files must already be colour. Sorting is by filename. This skips a
    video encode, which would quantise the colour and leave the file untagged.
    """
    if frames is not None:
        paths = sorted(
            path
            for path in Path(frames).iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not paths:
            raise SystemExit(f"no images in {frames}")
        return {"paths": paths, "index": 0, "camera": camera, "repeat": 1}
    if video is None:
        raise SystemExit("pass --video or --frames")
    return {"capture": cv2.VideoCapture(str(video)), "camera": camera, "repeat": repeat}


def load_twinlite(code, weights, config):
    """Lane lines and drivable area. The vehicle boxes stay with the other model."""
    import sys
    import types

    root = str(Path(code))
    if root not in sys.path:
        sys.path.insert(0, root)
    if "matplotlib" not in sys.modules:
        stub = types.ModuleType("matplotlib")
        stub.pyplot = types.ModuleType("matplotlib.pyplot")
        sys.modules["matplotlib"] = stub
        sys.modules["matplotlib.pyplot"] = stub.pyplot
    from model.model import TwinLiteNetPlus

    network = TwinLiteNetPlus(types.SimpleNamespace(config=config))
    state = torch.load(str(weights), map_location="cpu")
    network.load_state_dict(state)
    network.eval()
    return network


def twinlite_masks(network, image):
    """Return the lane mask and the drivable-area mask at the image size."""
    height, width = image.shape[:2]
    scale = min(640 / height, 640 / width)
    resized = (int(round(width * scale)), int(round(height * scale)))
    pad_w = float(np.mod(640 - resized[0], 32)) / 2
    pad_h = float(np.mod(640 - resized[1], 32)) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    canvas = cv2.resize(image, resized)
    canvas = cv2.copyMakeBorder(canvas, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = (
        torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)))
        .float()
        .div(255.0)
        .unsqueeze(0)
    )
    with torch.no_grad():
        drivable, lanes = network(tensor)
    bottom_end = drivable.shape[2] - bottom if bottom else drivable.shape[2]
    right_end = drivable.shape[3] - right if right else drivable.shape[3]
    drivable = drivable[:, :, top:bottom_end, left:right_end]
    lanes = lanes[:, :, top:bottom_end, left:right_end]
    drivable = torch.nn.functional.interpolate(drivable, size=(height, width), mode="bilinear", align_corners=False)
    lanes = torch.nn.functional.interpolate(lanes, size=(height, width), mode="bilinear", align_corners=False)
    return lanes.argmax(1)[0].cpu().numpy() == 1, drivable.argmax(1)[0].cpu().numpy() == 1


def rim_lines(mask, camera):
    """Left and right boundary of the drivable area, lifted onto the ground."""
    left, right = [], []
    lines = []

    def flush():
        nonlocal left, right
        for chain in (left, right):
            chain_np = np.asarray(chain)
            if len(chain_np) >= 6 and np.linalg.norm(np.diff(chain_np, axis=0), axis=1).sum() >= 4.0:
                lines.append(chain_np[::2])
        left, right = [], []

    previous = None
    for row in range(int(camera.cy) + 6, camera.height - 8, 3):
        cols = np.flatnonzero(mask[row])
        if len(cols) < 10 or (previous is not None and row - previous > 6):
            flush()
            previous = None
            continue
        if cols[0] > 2:
            point = camera.lift(float(cols[0]), float(row))
            if point is not None:
                left.append(point[:2])
        if cols[-1] < camera.width - 3:
            point = camera.lift(float(cols[-1]), float(row))
            if point is not None:
                right.append(point[:2])
        previous = row
    flush()
    return lines


def read_frame(source):
    camera = source["camera"]
    if "paths" in source:
        if source["index"] >= len(source["paths"]):
            return None
        image = cv2.imread(str(source["paths"][source["index"]]), cv2.IMREAD_COLOR)
        source["index"] += 1
        if image is None:
            raise SystemExit(f"could not read {source['paths'][source['index'] - 1]}")
    else:
        ok, image = source["capture"].read()
        if not ok:
            return None
        for _ in range(source["repeat"] - 1):
            source["capture"].read()
    if image.shape[1] != camera.width or image.shape[0] != camera.height:
        image = cv2.resize(image, (camera.width, camera.height))
    return image


def infer(source, weights, samples, camera, cache, lane_backend="yolopv2", lane_code=None, lane_weights=None, lane_config="large", prior_boxes=None):
    torch.set_num_threads(4)
    # The lane mask comes from the same forward pass as the boxes. Reusing boxes
    # still has to run that pass, unless a second model is supplying the lanes.
    model = None
    if prior_boxes is None or lane_backend != "twinlite":
        model = torch.jit.load(str(weights), map_location="cpu").eval()
    lanes_model = load_twinlite(lane_code, lane_weights, lane_config) if lane_backend == "twinlite" else None
    boxes_out, scores_out, lanes_out, edges_out = [], [], [], []
    for index in range(len(samples)):
        image = read_frame(source)
        if image is None:
            break
        mask = None
        if model is not None:
            content = cv2.resize(image, (MODEL, CONTENT_H))
            canvas = np.full((MODEL, MODEL, 3), 114, np.uint8)
            canvas[PAD_Y : PAD_Y + CONTENT_H, :] = content
            tensor = (
                torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)))
                .float()
                .div(255.0)
                .unsqueeze(0)
            )
            with torch.no_grad():
                pred, _drive, lanes = model(tensor)
            if lanes_model is None:
                band = (lanes[0, 0, PAD_Y : PAD_Y + CONTENT_H] > 0.5).cpu().numpy().astype(np.uint8)
                mask = cv2.resize(band, (camera.width, camera.height), interpolation=cv2.INTER_NEAREST) > 0
            if prior_boxes is None:
                image_boxes = decode_boxes(pred, camera)
                ego_boxes, ego_scores = [], []
                for x1, y1, x2, y2, score in image_boxes:
                    point = camera.lift((x1 + x2) / 2, y2)
                    if point is None:
                        continue
                    ego_boxes.append([x1, y1, x2, y2, point[0], point[1], score])
                    ego_scores.append(score)
                boxes_out.append(np.asarray(ego_boxes, dtype=np.float32).reshape(-1, 7))
                scores_out.append(np.asarray(ego_scores, dtype=np.float32))
        if prior_boxes is not None:
            boxes_out.append(np.asarray(prior_boxes[index], dtype=np.float32).reshape(-1, 7))
            row = np.asarray(prior_boxes[index], dtype=np.float32).reshape(-1, 7)
            scores_out.append(row[:, 6] if len(row) else np.zeros(0, np.float32))
        if lanes_model is not None:
            mask, drivable = twinlite_masks(lanes_model, image)
            edges = rim_lines(drivable, camera)
        else:
            edges = []
        lanes_out.append(lane_polylines(mask, camera))
        edges_out.append(edges)
        if index % 15 == 0:
            print(index, len(boxes_out[-1]), len(lanes_out[-1]), len(edges), flush=True)
    if len(boxes_out) != len(samples):
        raise SystemExit(f"video gave {len(boxes_out)} frames, motion file has {len(samples)}")
    np.savez_compressed(
        cache,
        boxes=np.array(boxes_out, dtype=object),
        scores=np.array(scores_out, dtype=object),
        lanes=np.array(lanes_out, dtype=object),
        edges=np.array(edges_out, dtype=object),
        lane_space=np.array("image"),
    )
    print("cached", len(boxes_out), flush=True)


def build_frames(samples, poses, cache, camera):
    data = np.load(cache, allow_pickle=True)
    packed = []
    for index, (sample, pose) in enumerate(zip(samples, poses)):
        raw = kept_boxes(data["boxes"][index])
        boxes, scores, labels = [], [], []
        for row in raw:
            point = ground_contact(camera, float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            if point is None:
                continue
            boxes.append([float(point[0]), float(point[1]), 0.0, 1.8, 4.2, 1.5, 0.0, 0.0, 0.0])
            scores.append(float(row[6]))
            labels.append(0)
        packed.append(
            {
                "token": str(index),
                "t": sample["t"],
                "pose": pose,
                "boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 9),
                "scores": np.asarray(scores, dtype=np.float64),
                "labels": np.asarray(labels, dtype=np.int64),
                "maps": {
                    "road_line": list(data["lanes"][index]),
                    "road_edge": [],
                },
                "image_boxes": raw,
            }
        )
    return packed


def colour(track_id):
    hue = (track_id * 47) % 180
    chip = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(chip, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_bev(objects, lanes, pose, width, height, now, wild=None, camera=None):
    scale = 12.0
    panel = np.full((height, width, 3), 28, np.uint8)
    origin_u = width // 2
    origin_v = int(height * 0.88)

    def screen(point):
        return int(origin_u - point[1] * scale), int(origin_v - point[0] * scale)

    for metre in range(-8, 56, 8):
        y = int(origin_v - metre * scale)
        if 0 <= y < height:
            cv2.line(panel, (0, y), (width, y), (48, 48, 48), 1)
            cv2.putText(panel, f"{metre}", (8, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
    for track in lanes:
        if track["hits"] < 4:
            continue
        local = world_to_ego(np.asarray(track["world"], dtype=np.float64), pose)
        if camera is not None:
            projected = camera.project_many(local)
            visible = (
                np.isfinite(projected[:, 0])
                & (projected[:, 0] > -40)
                & (projected[:, 0] < camera.width + 40)
                & (projected[:, 1] > -40)
                & (projected[:, 1] < camera.height + 40)
            )
        else:
            visible = np.ones(len(local), dtype=bool)
        segments = []
        current = []
        for point, show in zip(local, visible):
            if not show:
                if len(current) >= 2:
                    segments.append(current)
                current = []
                continue
            u, v = screen(point)
            if -40 <= u < width + 40 and -40 <= v < height + 40:
                current.append((u, v))
            else:
                if len(current) >= 2:
                    segments.append(current)
                current = []
        if len(current) >= 2:
            segments.append(current)
        paint = (40, 140, 255) if track.get("name") == "road_edge" else (80, 210, 90)
        for current in segments:
            cv2.polylines(panel, [np.asarray(current, np.int32)], False, paint, 2)
    for track in objects:
        if track["hits"] < 3 or track["name"] not in ("vehicle", "bicycle", "pedestrian"):
            continue
        trail = []
        last_t = None
        history = [sample for sample in track["samples"] if int(sample["token"]) <= now][-40:]
        for sample in history:
            local = world_to_ego(np.asarray(sample["world"], dtype=np.float64).reshape(1, 2), pose)[0]
            trail.append(screen(local))
            last_t = sample["token"]
        if len(trail) >= 2:
            cv2.polylines(panel, [np.asarray(trail, np.int32)], False, colour(track["id"]), 2)
        if last_t is None or abs(int(last_t) - now) > 2:
            continue
        u, v = trail[-1]
        if 0 <= u < width and 0 <= v < height:
            cv2.circle(panel, (u, v), 9, (0, 0, 0), -1)
            cv2.circle(panel, (u, v), 6, colour(track["id"]), -1)
            cv2.putText(panel, str(track["id"]), (u + 6, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
    for det in wild or []:
        centre = det.get("center_ego")
        if not centre:
            continue
        u, v = screen((centre[0], centre[1]))
        if not (-20 <= u < width + 20 and -20 <= v < height + 20):
            continue
        cv2.drawMarker(panel, (int(u), int(v)), (0, 200, 255), cv2.MARKER_DIAMOND, 14, 2)
        cv2.putText(panel, det["name"], (int(u) + 8, int(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    nose = screen((2.2, 0))
    left = screen((-0.6, 0.8))
    right = screen((-0.6, -0.8))
    cv2.fillConvexPoly(panel, np.array([nose, left, right]), (230, 230, 230))
    return panel


def render(source, output, samples, poses, objects, lanes, cache, camera, frame_dt, wild_frames=None):
    by_token = {}
    for track in objects:
        for sample in track["samples"]:
            by_token.setdefault(sample["token"], []).append((track["id"], sample["box"]))
    width, height = camera.width + 640, camera.height
    writer = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{1.0 / frame_dt:.6f}", "-i", "-",
            "-an", "-vf", "scale=in_range=full:out_range=tv:in_color_matrix=bt709:out_color_matrix=bt709",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    data = np.load(cache, allow_pickle=True)
    for index, pose in enumerate(poses):
        image = read_frame(source)
        if image is None:
            break
        for track in lanes:
            local = world_to_ego(np.asarray(track["world"], dtype=np.float64), pose)
            pixels = camera.project_many(local)
            current = []
            segments = []
            for pixel in pixels:
                if not np.isfinite(pixel[0]):
                    if len(current) >= 2:
                        segments.append(current)
                    current = []
                    continue
                u, v = int(pixel[0]), int(pixel[1])
                if -40 <= u < camera.width + 40 and -40 <= v < camera.height + 40:
                    current.append((u, v))
                else:
                    if len(current) >= 2:
                        segments.append(current)
                    current = []
            if len(current) >= 2:
                segments.append(current)
            for segment in segments:
                cv2.polylines(image, [np.asarray(segment, np.int32)], False, (40, 220, 60), 2)
        matched = by_token.get(str(index), [])
        for row in kept_boxes(data["boxes"][index]):
            point = ground_contact(camera, float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            if point is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in row[:4]]
            ident = None
            for track_id, box in matched:
                if abs(box[0] - point[0]) < 0.4 and abs(box[1] - point[1]) < 0.4:
                    ident = track_id
                    break
            paint = colour(ident) if ident is not None else (0, 190, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), paint, 2)
            if ident is not None:
                cv2.putText(image, str(ident), (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, paint, 1)
        wild = [] if wild_frames is None else wild_frames[index]["detections"]
        for det in wild:
            xyxy = det.get("xyxy")
            if not xyxy or len(xyxy) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy[:4]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = det["name"]
            centre = det.get("center_ego")
            guess = ground_contact(camera, x1, y1, x2, y2)
            if centre and guess is not None:
                label = f"{label} {centre[0] - guess[0]:+.1f},{centre[1] - guess[1]:+.1f}"
            cv2.putText(image, label, (x1, min(camera.height - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        panel = draw_bev(objects, lanes, pose, 640, camera.height, index, wild, camera)
        writer.stdin.write(np.concatenate([image, panel], axis=1).tobytes())
        if index % 30 == 0:
            print("render", index, flush=True)
    writer.stdin.close()
    code = writer.wait()
    if code != 0:
        raise SystemExit(f"ffmpeg exited {code}")
    print("wrote", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Front view and bird's-eye overlay for one clip.")
    parser.add_argument("--video", type=Path, help="Colour video, one view. Omit if --frames is set.")
    parser.add_argument("--frames", type=Path, help="Directory of colour images, sorted by name, one per motion row.")
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--camera", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path, help="TorchScript YOLOPv2, 1x3x640x640 input")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, help="Detection cache. Reused if it exists.")
    parser.add_argument("--repeat", type=int, default=1, help="Video frames per motion row. Use 2 when each frame is stored twice.")
    parser.add_argument("--lane-backend", choices=("yolopv2", "twinlite"), default="yolopv2")
    parser.add_argument("--lane-code", type=Path, help="TwinLiteNetPlus checkout. Required for --lane-backend twinlite.")
    parser.add_argument("--lane-weights", type=Path, help="TwinLiteNetPlus large.pth.")
    parser.add_argument("--lane-config", default="large", choices=("nano", "small", "medium", "large"))
    parser.add_argument("--reuse-boxes", type=Path, help="Keep the vehicle boxes in this cache and recompute the lanes.")
    parser.add_argument("--wilddet", type=Path, help="JSON from scripts/wilddet_clip.py. Drawn on top of the existing tracks.")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    camera = CylinderCamera.load(args.camera)
    samples, poses, frame_dt = load_motion(args.motion)
    cache = args.cache or args.output.with_suffix(".npz")
    if args.lane_backend == "twinlite" and (args.lane_code is None or args.lane_weights is None):
        raise SystemExit("--lane-backend twinlite needs --lane-code and --lane-weights")
    prior = None
    if args.reuse_boxes is not None:
        prior = np.load(args.reuse_boxes, allow_pickle=True)["boxes"]
        if len(prior) != len(samples):
            raise SystemExit(f"box cache has {len(prior)} frames, motion file has {len(samples)}")
    source = open_frames(args.video, args.frames, camera, args.repeat)
    image_lanes = False
    cached_edges = False
    if cache.exists():
        with np.load(cache, allow_pickle=True) as cached:
            cached_edges = "edges" in cached.files
            if "lane_space" in cached.files:
                image_lanes = str(cached["lane_space"].reshape(-1)[0]) == "image"
    fresh = (
        args.recompute
        or not cache.exists()
        or not image_lanes
        or (args.lane_backend == "twinlite" and not cached_edges)
    )
    if fresh:
        infer(
            source, args.weights, samples, camera, cache,
            lane_backend=args.lane_backend, lane_code=args.lane_code, lane_weights=args.lane_weights,
            lane_config=args.lane_config, prior_boxes=prior,
        )
        source = open_frames(args.video, args.frames, camera, args.repeat)
    packed = build_frames(samples, poses, cache, camera)
    objects = track_objects(packed, score_min=0.4, max_gap=max(8, int(round(1.0 / frame_dt))))
    smooth_world(objects, poses, frame_dt)
    extend_tracks(objects, packed, horizon=2.0)
    lanes = fit_lane_curves(packed, camera.project_many, camera.lift)
    print(
        "tracks", len(objects),
        "lanes", len(lanes),
        "lane_px", [track.get("error_px") for track in lanes],
        "dt", round(frame_dt, 4),
        flush=True,
    )
    wild_frames = None
    if args.wilddet is not None:
        wild_frames = json.loads(args.wilddet.read_text())["frames"]
        if len(wild_frames) != len(samples):
            raise SystemExit(f"wilddet file has {len(wild_frames)} frames, motion file has {len(samples)}")
    render(source, args.output, samples, poses, objects, lanes, cache, camera, frame_dt, wild_frames)


if __name__ == "__main__":
    main()
