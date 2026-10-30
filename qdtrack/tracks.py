"""Object and lane tracking on top of single-frame Qwen-Drive outputs.

Perception is one frame at a time. Tracking stitches those frames in a local
odometry frame built from the vehicle bus (speed + yaw rate).
"""

from __future__ import annotations

import numpy as np

DET_CLASS_NAMES = (
    "vehicle",
    "czone_sign",
    "bicycle",
    "generic_object",
    "pedestrian",
    "traffic_cone",
    "barrier",
)
MAP_CLASS_NAMES = (
    "background",
    "driveable_surface",
    "road_line",
    "road_edge",
    "crosswalk",
    "walkway",
)
LANE_CLASSES = ("road_line", "road_edge")
MAP_XBOUND = (-30.0, 30.0, 0.15)
MAP_YBOUND = (-15.0, 15.0, 0.15)

# Greedy gate in metres. Pedestrians and cones move less between 5 Hz frames.
_GATE = {
    "vehicle": 6.0,
    "bicycle": 3.5,
    "pedestrian": 2.2,
    "traffic_cone": 1.6,
    "barrier": 2.0,
    "czone_sign": 2.0,
    "generic_object": 3.0,
}


def _yaw_of(pose: dict) -> float:
    return float(pose["yaw"])


def ego_to_world(points: np.ndarray, pose: dict) -> np.ndarray:
    """points (N, 2) or (N, 3) in ego -> world XY (and Z unchanged if present)."""
    yaw = _yaw_of(pose)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    xy = points[:, :2] @ rotation.T
    xy[:, 0] += pose["x"]
    xy[:, 1] += pose["y"]
    if points.shape[1] == 2:
        return xy
    out = np.concatenate([xy, points[:, 2:3]], axis=1)
    return out


def world_to_ego(points: np.ndarray, pose: dict) -> np.ndarray:
    yaw = _yaw_of(pose)
    c, s = np.cos(yaw), np.sin(yaw)
    delta = points[:, :2] - np.array([pose["x"], pose["y"]], dtype=np.float64)
    rotation_t = np.array([[c, s], [-s, c]], dtype=np.float64)
    xy = delta @ rotation_t.T
    if points.shape[1] == 2:
        return xy
    return np.concatenate([xy, points[:, 2:3]], axis=1)


def integrate_ego(samples: list[dict]) -> list[dict]:
    """Integrate speed (m/s) and yaw rate (rad/s) into a local pose.

    samples are time-ordered dicts with t (seconds), speed_mps, yaw_rate.
    The first pose is the origin, heading 0, x forward of that instant.
    """
    poses = []
    x = y = yaw = 0.0
    prev_t = None
    for sample in samples:
        t = float(sample["t"])
        if prev_t is not None:
            dt = t - prev_t
            # Mid-step speed, yaw rate applied before the translation so a
            # turn does not lag the chord by a full frame.
            yaw = yaw + float(sample["yaw_rate"]) * dt
            x = x + float(sample["speed_mps"]) * np.cos(yaw) * dt
            y = y + float(sample["speed_mps"]) * np.sin(yaw) * dt
        prev_t = t
        poses.append({"x": float(x), "y": float(y), "yaw": float(yaw), "t": t})
    return poses


def _as_detections(frame: dict, score_min: float) -> list[dict]:
    boxes = np.asarray(frame.get("boxes", []), dtype=np.float64).reshape(-1, 9)
    scores = np.asarray(frame.get("scores", []), dtype=np.float64).reshape(-1)
    labels = np.asarray(frame.get("labels", []), dtype=np.int64).reshape(-1)
    out = []
    n = min(len(boxes), len(scores), len(labels))
    for i in range(n):
        if scores[i] < score_min:
            continue
        label = int(labels[i])
        if label < 0 or label >= len(DET_CLASS_NAMES):
            continue
        out.append(
            {
                "box": boxes[i].copy(),
                "score": float(scores[i]),
                "label": label,
                "name": DET_CLASS_NAMES[label],
            }
        )
    return out


def track_objects(frames: list[dict], score_min: float = 0.25) -> list[dict]:
    """Nearest-first constant-velocity tracker. frames carry pose + boxes in ego.

    box is [x, y, z, w, l, h, yaw, vx, vy] in the ego frame (lidar == ego).
    Pairs inside the class gate are assigned shortest-distance first so a
    nearby track cannot steal a detection that belongs to someone else.
    """
    tracks: list[dict] = []
    next_id = 1
    for index, frame in enumerate(frames):
        pose = frame["pose"]
        detections = _as_detections(frame, score_min)
        predictions = []
        for track in tracks:
            dt = frame["t"] - track["last_t"]
            predictions.append((track["world"] + track["vel_world"] * dt, dt))
        candidates = []
        for track_index, track in enumerate(tracks):
            if index - track["last_index"] > 3:
                continue
            predicted, _dt = predictions[track_index]
            gate = _GATE.get(track["name"], 3.0)
            for j, det in enumerate(detections):
                if det["name"] != track["name"]:
                    continue
                centre = ego_to_world(det["box"][None, :2], pose)[0]
                dist = float(np.linalg.norm(centre - predicted))
                if dist <= gate:
                    candidates.append((dist, track_index, j, centre))
        candidates.sort(key=lambda item: item[0])
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _dist, track_index, j, centre in candidates:
            if track_index in used_tracks or j in used_dets:
                continue
            used_tracks.add(track_index)
            used_dets.add(j)
            track = tracks[track_index]
            det = detections[j]
            dt = frame["t"] - track["last_t"]
            if dt > 1e-3:
                vel = (centre - track["world"]) / dt
            else:
                vel = track["vel_world"]
            yaw = _yaw_of(pose)
            c, s = np.cos(yaw), np.sin(yaw)
            model_v = np.array(
                [c * det["box"][7] - s * det["box"][8], s * det["box"][7] + c * det["box"][8]],
                dtype=np.float64,
            )
            vel = 0.5 * vel + 0.5 * model_v
            track["world"] = centre
            track["vel_world"] = vel
            track["last_t"] = frame["t"]
            track["last_index"] = index
            track["last_score"] = det["score"]
            track["hits"] += 1
            track["samples"].append(
                {
                    "token": frame["token"],
                    "score": det["score"],
                    "box": [round(float(v), 4) for v in det["box"]],
                    "world": [round(float(centre[0]), 3), round(float(centre[1]), 3)],
                }
            )
        for j, det in enumerate(detections):
            if j in used_dets:
                continue
            centre = ego_to_world(det["box"][None, :2], pose)[0]
            yaw = _yaw_of(pose)
            c, s = np.cos(yaw), np.sin(yaw)
            vel = np.array(
                [c * det["box"][7] - s * det["box"][8], s * det["box"][7] + c * det["box"][8]],
                dtype=np.float64,
            )
            tracks.append(
                {
                    "id": next_id,
                    "name": det["name"],
                    "label": det["label"],
                    "world": centre,
                    "vel_world": vel,
                    "last_t": frame["t"],
                    "last_index": index,
                    "last_score": det["score"],
                    "hits": 1,
                    "samples": [
                        {
                            "token": frame["token"],
                            "score": det["score"],
                            "box": [round(float(v), 4) for v in det["box"]],
                            "world": [round(float(centre[0]), 3), round(float(centre[1]), 3)],
                        }
                    ],
                }
            )
            next_id += 1
    published = []
    for track in tracks:
        if track["hits"] < 2 and track["last_score"] < 0.45:
            continue
        published.append(
            {
                "id": track["id"],
                "name": track["name"],
                "label": track["label"],
                "hits": track["hits"],
                "samples": track["samples"],
            }
        )
    return published


def _bounds_to_size(bound: tuple[float, float, float]) -> int:
    return int(round((bound[1] - bound[0]) / bound[2]))


def _pixel_to_ego(cols: np.ndarray, rows: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Map raster pixels to ego XY. Accepts either (nx, ny) or (ny, nx)."""
    nx = _bounds_to_size(MAP_XBOUND)
    ny = _bounds_to_size(MAP_YBOUND)
    height, width = shape
    if (height, width) == (nx, ny):
        x = MAP_XBOUND[0] + (rows.astype(np.float64) + 0.5) * MAP_XBOUND[2]
        y = MAP_YBOUND[0] + (cols.astype(np.float64) + 0.5) * MAP_YBOUND[2]
    elif (height, width) == (ny, nx):
        y = MAP_YBOUND[0] + (rows.astype(np.float64) + 0.5) * MAP_YBOUND[2]
        x = MAP_XBOUND[0] + (cols.astype(np.float64) + 0.5) * MAP_XBOUND[2]
    else:
        # Unknown resolution: treat rows as x spanning the bound, cols as y.
        x = MAP_XBOUND[0] + (rows.astype(np.float64) + 0.5) / height * (MAP_XBOUND[1] - MAP_XBOUND[0])
        y = MAP_YBOUND[0] + (cols.astype(np.float64) + 0.5) / width * (MAP_YBOUND[1] - MAP_YBOUND[0])
    return np.stack([x, y], axis=1)


def _skeleton(mask: np.ndarray) -> np.ndarray:
    import cv2

    img = (mask.astype(np.uint8) * 255)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel > 0


def _trace(skel: np.ndarray) -> list[np.ndarray]:
    """Walk a skeleton into polylines. Returns list of (N, 2) row/col arrays."""
    ys, xs = np.where(skel)
    if len(xs) == 0:
        return []
    nodes = {(int(y), int(x)) for y, x in zip(ys, xs)}
    neighbours = {}
    for y, x in nodes:
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                p = (y + dy, x + dx)
                if p in nodes:
                    nbrs.append(p)
        neighbours[(y, x)] = nbrs
    endpoints = [p for p, n in neighbours.items() if len(n) != 2]
    starts = endpoints or [next(iter(nodes))]
    seen = set()
    lines = []
    for start in starts:
        if start in seen:
            continue
        path = [start]
        seen.add(start)
        prev = None
        cur = start
        while True:
            nxts = [n for n in neighbours[cur] if n != prev and n not in seen]
            if not nxts:
                break
            nxt = nxts[0]
            path.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        if len(path) >= 8:
            lines.append(np.array(path, dtype=np.int32))
    # Leftover loops.
    leftover = [p for p in nodes if p not in seen]
    if len(leftover) >= 8:
        lines.append(np.array(leftover, dtype=np.int32))
    return lines


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    segment = end - start
    length = np.linalg.norm(segment) + 1e-9
    dist = np.abs((points[:, 0] - start[0]) * segment[1] - (points[:, 1] - start[1]) * segment[0]) / length
    index = int(np.argmax(dist))
    if dist[index] > epsilon:
        left = _rdp(points[: index + 1], epsilon)
        right = _rdp(points[index:], epsilon)
        return np.concatenate([left[:-1], right], axis=0)
    return np.stack([start, end])


def polylines_from_map(label_map: np.ndarray, class_name: str, min_points: int = 12) -> list[np.ndarray]:
    class_id = MAP_CLASS_NAMES.index(class_name)
    mask = label_map == class_id
    if int(mask.sum()) < min_points:
        return []
    import cv2

    num, cc = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    polylines = []
    for cid in range(1, num):
        component = cc == cid
        if int(component.sum()) < min_points:
            continue
        skel = _skeleton(component)
        if int(skel.sum()) < 8:
            skel = component
        for path in _trace(skel):
            rows, cols = path[:, 0], path[:, 1]
            ego = _pixel_to_ego(cols, rows, label_map.shape)
            ego = _rdp(ego, 0.4)
            if len(ego) >= 2 and _polyline_length(ego) >= 2.0:
                polylines.append(ego)
    return polylines


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 1e9
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def track_lanes(frames: list[dict], gate: float = 1.6) -> list[dict]:
    """Associate per-frame lane polylines (ego XY) into world tracks.

    frames need pose and maps[name] = list of (N, 2) ego polylines.
    """
    tracks: list[dict] = []
    next_id = 1
    for index, frame in enumerate(frames):
        pose = frame["pose"]
        for name in LANE_CLASSES:
            for poly in frame.get("maps", {}).get(name, []):
                poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
                world = ego_to_world(poly, pose)
                best = None
                best_dist = gate
                for track in tracks:
                    if track["name"] != name or index - track["last_index"] > 2:
                        continue
                    dist = _chamfer(world, track["world"])
                    if dist < best_dist:
                        best_dist = dist
                        best = track
                if best is None:
                    tracks.append(
                        {
                            "id": next_id,
                            "name": name,
                            "world": world,
                            "last_index": index,
                            "hits": 1,
                            "samples": [
                                {
                                    "token": frame["token"],
                                    "ego": np.round(poly, 3).tolist(),
                                }
                            ],
                        }
                    )
                    next_id += 1
                else:
                    fused = np.concatenate([best["world"], world], axis=0)
                    # Keep a stable centreline: voxel-downsample in world.
                    key = np.round(fused / 0.5).astype(np.int32)
                    _, unique = np.unique(key, axis=0, return_index=True)
                    fused = fused[np.sort(unique)]
                    # Reorder along the long axis so the viewer gets a line, not a cloud.
                    axis = int(np.argmax(fused.max(0) - fused.min(0)))
                    fused = fused[np.argsort(fused[:, axis])]
                    best["world"] = fused
                    best["last_index"] = index
                    best["hits"] += 1
                    best["samples"].append({"token": frame["token"], "ego": np.round(poly, 3).tolist()})
    published = []
    for track in tracks:
        if track["hits"] < 2 and _polyline_length(track["world"]) < 4:
            continue
        published.append(
            {
                "id": track["id"],
                "name": track["name"],
                "hits": track["hits"],
                "world": np.round(track["world"], 3).tolist(),
                "samples": track["samples"],
            }
        )
    return _collapse_lanes(published)


def _collapse_lanes(tracks: list[dict]) -> list[dict]:
    """One physical marking is traced many times, a few tenths of a metre apart.

    Keep the straight pieces that follow the road, then stitch copies whose
    lateral offset agrees and whose ranges along the road meet or overlap.
    """
    kept = []
    for track in tracks:
        world = np.asarray(track["world"], dtype=np.float64).reshape(-1, 2)
        if len(world) < 2:
            continue
        length = _polyline_length(world)
        chord = float(np.linalg.norm(world[-1] - world[0]))
        angle = float(np.arctan2(world[-1, 1] - world[0, 1], world[-1, 0] - world[0, 0]))
        if angle > np.pi / 2:
            angle -= np.pi
        if angle < -np.pi / 2:
            angle += np.pi
        if length < 8 or chord / max(length, 1e-6) < 0.9 or abs(angle) > 0.35:
            continue
        kept.append(
            {
                "track": track,
                "length": length,
                "y": float(world[:, 1].mean()),
                "x0": float(world[:, 0].min()),
                "x1": float(world[:, 0].max()),
                "world": world,
            }
        )
    kept.sort(key=lambda item: (item["track"]["name"], item["y"]))
    groups: list[dict] = []
    for item in kept:
        group = None
        for candidate in groups:
            if candidate["name"] != item["track"]["name"]:
                continue
            if abs(candidate["y"] - item["y"]) >= 1.3:
                continue
            if item["x1"] < candidate["x0"] - 12 or candidate["x1"] < item["x0"] - 12:
                continue
            group = candidate
            break
        if group is None:
            groups.append(
                {
                    "name": item["track"]["name"],
                    "y": item["y"],
                    "weight": item["length"],
                    "hits": item["track"]["hits"],
                    "x0": item["x0"],
                    "x1": item["x1"],
                    "world": [item["world"]],
                    "samples": list(item["track"]["samples"]),
                }
            )
            continue
        group["y"] = (group["y"] * group["weight"] + item["y"] * item["length"]) / (group["weight"] + item["length"])
        group["weight"] += item["length"]
        group["hits"] += item["track"]["hits"]
        group["x0"] = min(group["x0"], item["x0"])
        group["x1"] = max(group["x1"], item["x1"])
        group["world"].append(item["world"])
        group["samples"].extend(item["track"]["samples"])
    collapsed = []
    for group in groups:
        if group["x1"] - group["x0"] < 15:
            continue
        pts = np.concatenate(group["world"], axis=0)
        pts = pts[np.argsort(pts[:, 0])]
        keep = [pts[0]]
        for point in pts[1:]:
            if point[0] - keep[-1][0] >= 1.5:
                keep.append(point)
        if len(keep) < 2:
            continue
        collapsed.append(
            {
                "id": len(collapsed) + 1,
                "name": group["name"],
                "hits": group["hits"],
                "world": np.round(np.stack(keep), 3).tolist(),
                "samples": group["samples"],
            }
        )
    return collapsed
