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
from qdtrack.tracks import integrate_ego, track_lanes, track_objects, world_to_ego

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
    """Centre of each lane-mask run, linked down the image, then lifted."""
    rows = []
    for v in range(int(camera.cy) + 6, camera.height - 8, 3):
        cols = np.flatnonzero(mask[v])
        if len(cols) == 0:
            rows.append([])
            continue
        splits = np.where(np.diff(cols) > 6)[0] + 1
        runs = np.split(cols, splits)
        points = []
        for run in runs:
            if not (2 <= len(run) <= 50):
                continue
            point = camera.lift(float(run.mean()), float(v))
            if point is not None:
                points.append(point[:2])
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
            for n in range(r + 1, min(r + 80, len(rows))):
                best, best_d = None, 1.2
                for j, cand in enumerate(rows[n]):
                    if j in used[n]:
                        continue
                    dist = abs(cand[1] - prev[1]) + 0.15 * abs(cand[0] - prev[0])
                    if dist < best_d and cand[0] > prev[0] - 1.0:
                        best, best_d = j, dist
                if best is None:
                    break
                used[n].add(best)
                prev = rows[n][best]
                chain.append(prev)
            chain = np.asarray(chain)
            if len(chain) >= 6 and np.linalg.norm(np.diff(chain, axis=0), axis=1).sum() >= 4.0:
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
    sigma_a = 1.5
    for index in range(count):
        dt = steps[index]
        motion = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        noise = sigma_a ** 2 * np.array(
            [
                [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                [dt ** 3 / 2, 0, dt ** 2, 0],
                [0, dt ** 3 / 2, 0, dt ** 2],
            ]
        )
        state = motion @ state
        covariance = motion @ covariance @ motion.T + noise
        cosine, sine = np.cos(headings[index]), np.sin(headings[index])
        axes = np.array([[cosine * 2.5, -sine * 0.6], [sine * 2.5, cosine * 0.6]])
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
    """Hold a car whose sideways motion is the viewing angle, not the car.

    The bottom of the box is the rear bumper from behind and a side corner
    once the car is abreast. If that sideways motion follows the viewing angle
    and the car has not moved along the road, keep the head-on position.
    A car that is closing at the ego speed is parked and is held the same way.
    Anything else keeps the Kalman speed.
    """
    for track in tracks:
        worlds = np.array([sample["world"] for sample in track["samples"]], dtype=np.float64)
        if len(worlds) < 3:
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
        first, last = track["samples"][0], track["samples"][-1]
        start, end = int(first["token"]), int(last["token"])
        driven = float(np.hypot(poses[end]["x"] - poses[start]["x"], poses[end]["y"] - poses[start]["y"]))
        closing = float(first["box"][0] - last["box"][0])
        parked = len(worlds) >= 8 and driven > 5 and abs(closing - driven) < max(6.0, 0.5 * driven)
        if passing or parked:
            path = np.repeat(reference.reshape(1, 2), len(worlds), axis=0)
        else:
            path = kalman_positions(track, poses, frame_dt)
        for sample, point in zip(track["samples"], path):
            sample["world"] = [float(point[0]), float(point[1])]


def read_frame(capture, camera, repeat):
    ok, image = capture.read()
    if not ok:
        return None
    for _ in range(repeat - 1):
        capture.read()
    if image.shape[1] != camera.width or image.shape[0] != camera.height:
        image = cv2.resize(image, (camera.width, camera.height))
    return image


def infer(video, weights, samples, camera, cache, repeat):
    torch.set_num_threads(4)
    model = torch.jit.load(str(weights), map_location="cpu").eval()
    capture = cv2.VideoCapture(str(video))
    boxes_out, scores_out, lanes_out = [], [], []
    for index in range(len(samples)):
        image = read_frame(capture, camera, repeat)
        if image is None:
            break
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
        band = (lanes[0, 0, PAD_Y : PAD_Y + CONTENT_H] > 0.5).cpu().numpy().astype(np.uint8)
        mask = cv2.resize(band, (camera.width, camera.height), interpolation=cv2.INTER_NEAREST) > 0
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
        lanes_out.append(lane_polylines(mask, camera))
        if index % 15 == 0:
            print(index, len(ego_boxes), len(lanes_out[-1]), flush=True)
    if len(boxes_out) != len(samples):
        raise SystemExit(f"video gave {len(boxes_out)} frames, motion file has {len(samples)}")
    np.savez_compressed(
        cache,
        boxes=np.array(boxes_out, dtype=object),
        scores=np.array(scores_out, dtype=object),
        lanes=np.array(lanes_out, dtype=object),
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
                "maps": {"road_line": list(data["lanes"][index]), "road_edge": []},
                "image_boxes": raw,
            }
        )
    return packed


def colour(track_id):
    hue = (track_id * 47) % 180
    chip = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(chip, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_bev(objects, lanes, pose, width, height, now):
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
        pts = []
        for point in local:
            u, v = screen(point)
            if -40 <= u < width + 40 and -40 <= v < height + 40:
                pts.append((u, v))
        if len(pts) >= 2:
            cv2.polylines(panel, [np.asarray(pts, np.int32)], False, (80, 210, 90), 2)
    for track in objects:
        if track["hits"] < 3 or track["name"] != "vehicle":
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
    nose = screen((2.2, 0))
    left = screen((-0.6, 0.8))
    right = screen((-0.6, -0.8))
    cv2.fillConvexPoly(panel, np.array([nose, left, right]), (230, 230, 230))
    return panel


def render(video, output, samples, poses, objects, lanes, cache, camera, repeat, frame_dt):
    by_token = {}
    for track in objects:
        for sample in track["samples"]:
            by_token.setdefault(sample["token"], []).append((track["id"], sample["box"]))
    capture = cv2.VideoCapture(str(video))
    width, height = camera.width + 640, camera.height
    writer = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{1.0 / frame_dt:.6f}", "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    data = np.load(cache, allow_pickle=True)
    for index, pose in enumerate(poses):
        image = read_frame(capture, camera, repeat)
        if image is None:
            break
        for line in data["lanes"][index]:
            pixels = []
            for x, y in line:
                pixel = camera.project(x, y)
                if pixel is None:
                    continue
                pixels.append((int(pixel[0]), int(pixel[1])))
            if len(pixels) >= 2:
                cv2.polylines(image, [np.asarray(pixels, np.int32)], False, (40, 220, 60), 2)
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
        panel = draw_bev(objects, lanes, pose, 640, camera.height, index)
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
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--camera", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path, help="TorchScript YOLOPv2, 1x3x640x640 input")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, help="Detection cache. Reused if it exists.")
    parser.add_argument("--repeat", type=int, default=1, help="Video frames per motion row. Use 2 when each frame is stored twice.")
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    camera = CylinderCamera.load(args.camera)
    samples, poses, frame_dt = load_motion(args.motion)
    cache = args.cache or args.output.with_suffix(".npz")
    if args.recompute or not cache.exists():
        infer(args.video, args.weights, samples, camera, cache, args.repeat)
    packed = build_frames(samples, poses, cache, camera)
    objects = track_objects(packed, score_min=0.4, max_gap=8)
    smooth_world(objects, poses, frame_dt)
    lanes = track_lanes(packed, gate=1.6, max_gap=3)
    print("tracks", len(objects), "lanes", len(lanes), "dt", round(frame_dt, 4), flush=True)
    render(args.video, args.output, samples, poses, objects, lanes, cache, camera, args.repeat, frame_dt)


if __name__ == "__main__":
    main()
