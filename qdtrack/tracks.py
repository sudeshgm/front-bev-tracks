"""Object and lane tracking on single-frame detections.

Detections are one frame at a time. Tracking stitches them in a local
odometry frame built from the vehicle speed and yaw rate.
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

# A missed detection can rejoin for about a second. Longer than that, the
# track ends. A sideways jump is rejected separately: the car beside this
# one can sit where the old speed predicts, with no gap at all.
_COAST_S = 1.0


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


def _vehicle_cost(predicted, box, pose) -> float:
    """Range is the noisy axis of a ground-contact point. Lateral is not.

    Five pixels of box-bottom error at this camera is about ``x**2 * 5/590``
    metres, so an oncoming car can jump several metres in range and still be
    the same car. A shift of one lane is not.
    """
    ahead = world_to_ego(np.asarray(predicted, dtype=np.float64).reshape(1, 2), pose)[0]
    along = float(box[0] - ahead[0])
    lateral = float(box[1] - ahead[1])
    distance = max(float(ahead[0]), 5.0)
    sigma_along = max(2.5, distance * distance * 5.0 / 590.0)
    return (along / sigma_along) ** 2 + (lateral / 0.8) ** 2


def track_objects(frames: list[dict], score_min: float = 0.25, max_gap: int = 3) -> list[dict]:
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
            if index - track["last_index"] > max_gap:
                continue
            if frame["t"] - track["last_t"] > _COAST_S:
                continue
            predicted, _dt = predictions[track_index]
            for j, det in enumerate(detections):
                if det["name"] != track["name"]:
                    continue
                centre = ego_to_world(det["box"][None, :2], pose)[0]
                if track["name"] == "vehicle":
                    previous = world_to_ego(np.asarray(track["world"], dtype=np.float64).reshape(1, 2), pose)[0]
                    elapsed = max(frame["t"] - track["last_t"], 1e-3)
                    sideways = abs(float(det["box"][1]) - float(previous[1]))
                    # Two metres in one frame at this camera is a different car.
                    # A real car does not move a lane sideways between frames.
                    if sideways > min(2.0 * max(elapsed / 0.07, 1.0), 3.5):
                        continue
                    cost = _vehicle_cost(predicted, det["box"], pose)
                    if cost > 9.0:
                        continue
                else:
                    cost = float(np.linalg.norm(centre - predicted))
                    if cost > _GATE.get(track["name"], 3.0):
                        continue
                candidates.append((cost, track_index, j, centre))
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
                step = (centre - track["world"]) / dt
            else:
                step = track["vel_world"]
            yaw = _yaw_of(pose)
            c, s = np.cos(yaw), np.sin(yaw)
            model_v = np.array(
                [c * det["box"][7] - s * det["box"][8], s * det["box"][7] + c * det["box"][8]],
                dtype=np.float64,
            )
            # A zero in the box is not a measured speed. One noisy step is not
            # the speed either: follow the steps, but keep most of the speed
            # the track already has.
            if float(np.linalg.norm(model_v)) > 0.5:
                vel = 0.5 * step + 0.5 * model_v
            elif track["hits"] >= 2 and float(np.linalg.norm(track["vel_world"])) > 0.3:
                vel = 0.6 * track["vel_world"] + 0.4 * step
            else:
                vel = step
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
    place_centers(published, frames)
    return extend_tracks(published, frames)


def place_centers(tracks: list[dict], frames: list[dict]) -> list[dict]:
    """Move a point from the near face to the centre, along the track heading.

    The ground contact is the face the camera sees. The centre is half a
    length further from the ego, along the way the track itself is moving.
    A track that has not moved has no heading, and is left on the face.
    """
    pose_of = {str(frame["token"]): frame["pose"] for frame in frames}
    for track in tracks:
        measured = [sample for sample in track["samples"] if not sample.get("predicted")]
        if len(measured) < 2:
            continue
        delta = np.asarray(measured[-1]["world"], dtype=np.float64) - np.asarray(measured[0]["world"], dtype=np.float64)
        length = float(np.linalg.norm(delta))
        if length < 1.0:
            continue
        heading = delta / length
        for sample in measured:
            pose = pose_of.get(str(sample["token"]))
            box = sample.get("box") or []
            if pose is None or len(box) < 5:
                continue
            half = 0.5 * float(box[4])
            point = np.asarray(sample["world"], dtype=np.float64)
            away = point - np.array([pose["x"], pose["y"]], dtype=np.float64)
            shift = heading * half if float(np.dot(heading, away)) >= 0 else -heading * half
            moved = point + shift
            sample["face"] = [round(float(point[0]), 3), round(float(point[1]), 3)]
            sample["world"] = [round(float(moved[0]), 3), round(float(moved[1]), 3)]
    return tracks


def extend_tracks(tracks: list[dict], frames: list[dict], horizon: float = 2.0) -> list[dict]:
    """Continue a moving track for up to two seconds after its last detection.

    The tag needs the crossing of the ego path. The car often leaves the
    image before that. The same speed the positions already support is
    integrated through the frames that follow, and those points are marked
    predicted. A car that is standing still is not invented forward.
    """
    if len(frames) < 2:
        return tracks
    ordered = sorted(frames, key=lambda frame: float(frame["t"]))
    times = [float(frame["t"]) for frame in ordered]
    tokens = [str(frame["token"]) for frame in ordered]
    index_of = {token: index for index, token in enumerate(tokens)}
    for track in tracks:
        measured = [sample for sample in track["samples"] if not sample.get("predicted")]
        track["samples"] = measured
        if len(measured) < 2:
            continue
        last = index_of.get(str(measured[-1]["token"]))
        early_at = max(0, len(measured) - 4)
        early = index_of.get(str(measured[early_at]["token"]))
        if last is None or early is None or last <= early:
            continue
        span = times[last] - times[early]
        if span < 1e-3:
            continue
        velocity = (
            np.asarray(measured[-1]["world"], dtype=np.float64)
            - np.asarray(measured[early_at]["world"], dtype=np.float64)
        ) / span
        if float(np.linalg.norm(velocity)) < 0.5:
            continue
        # Hold that velocity in the ego frame. A straight line in the world
        # turns away from the ego as soon as the ego turns, and the car stops
        # counting as oncoming. The speed stays the last measured speed.
        last_yaw = float(ordered[last]["pose"]["yaw"])
        cosine, sine = np.cos(last_yaw), np.sin(last_yaw)
        ahead = np.array(
            [cosine * velocity[0] + sine * velocity[1], -sine * velocity[0] + cosine * velocity[1]],
            dtype=np.float64,
        )
        point = np.asarray(measured[-1]["world"], dtype=np.float64)
        elapsed = 0.0
        last_box = measured[-1].get("box")
        last_score = measured[-1]["score"]
        for index in range(last + 1, len(ordered)):
            step = times[index] - times[index - 1]
            if step <= 0:
                continue
            elapsed += step
            if elapsed > horizon:
                break
            yaw = float(ordered[index]["pose"]["yaw"])
            cosine, sine = np.cos(yaw), np.sin(yaw)
            step_world = np.array(
                [cosine * ahead[0] - sine * ahead[1], sine * ahead[0] + cosine * ahead[1]],
                dtype=np.float64,
            )
            point = point + step_world * step
            measured.append(
                {
                    "token": tokens[index],
                    "score": last_score,
                    "box": last_box,
                    "world": [round(float(point[0]), 3), round(float(point[1]), 3)],
                    "predicted": True,
                }
            )
    return tracks


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


def _overlap_distance(new: np.ndarray, old: np.ndarray) -> float:
    """Mean distance from the new points to the overlapping part of the track.

    A symmetric chamfer against the whole history grows with the road already
    driven, so one marking was split every few dozen metres. Only the part of
    the track that sits next to the new segment is compared.
    """
    if len(new) == 0 or len(old) == 0:
        return 1e9
    lo = new.min(axis=0) - 4.0
    hi = new.max(axis=0) + 4.0
    window = old[np.all((old >= lo) & (old <= hi), axis=1)]
    if len(window) == 0:
        ends = np.stack([old[0], old[-1]])
        return float(np.linalg.norm(new[:, None, :] - ends[None, :, :], axis=2).min())
    dist = np.linalg.norm(new[:, None, :] - window[None, :, :], axis=2)
    # A parallel marking is about 3 m away. A cloud must not swallow it just
    # because some of the cloud's points happen to lie within the distance gate.
    if abs(float(np.median(new[:, 1]) - np.median(window[:, 1]))) > 0.8:
        return 1e9
    return float(dist.min(axis=1).mean())


def track_lanes(frames: list[dict], gate: float = 1.6, max_gap: int = 2) -> list[dict]:
    """Associate per-frame lane polylines (ego XY) into world tracks.

    frames need pose and maps[name] = list of (N, 2) ego polylines.
    This keeps every trace. The overlay does not use it; a marking there is one
    curve fitted by reprojection, not a chain of these traces.
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
                    if track["name"] != name or index - track["last_index"] > max_gap:
                        continue
                    dist = _overlap_distance(world, track["world"])
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
                    key = np.round(fused / 0.5).astype(np.int32)
                    _, unique = np.unique(key, axis=0, return_index=True)
                    fused = fused[np.sort(unique)]
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
    return published


# A chain joins a curve's seed only when its ground estimate is this close.
# Parallel markings are about 3.5 m apart, so they stay separate. The number
# does not set the curve: the reprojection fit does.
_GROUP_M = 1.5
_PIXEL_GATE = 12.0
_MIN_FRAMES = 4
_MIN_SPAN_M = 6.0


def _path_from_poses(poses: list[dict]):
    xy = np.array([[pose["x"], pose["y"]] for pose in poses], dtype=np.float64)
    yaw = np.array([pose["yaw"] for pose in poses], dtype=np.float64)
    if len(xy) == 1:
        return xy, np.array([0.0]), yaw
    keep = [0]
    for index in range(1, len(xy)):
        if np.linalg.norm(xy[index] - xy[keep[-1]]) >= 0.05:
            keep.append(index)
    if keep[-1] != len(xy) - 1:
        keep.append(len(xy) - 1)
    xy = xy[keep]
    yaw = yaw[keep]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    station = np.concatenate([[0.0], np.cumsum(step)])
    return xy, station, yaw


def _pose_at(station, path_xy, path_s, path_yaw):
    if station <= path_s[0]:
        yaw = float(path_yaw[0])
        offset = station - path_s[0]
        return path_xy[0] + offset * np.array([np.cos(yaw), np.sin(yaw)]), yaw
    if station >= path_s[-1]:
        yaw = float(path_yaw[-1])
        offset = station - path_s[-1]
        return path_xy[-1] + offset * np.array([np.cos(yaw), np.sin(yaw)]), yaw
    index = int(np.searchsorted(path_s, station) - 1)
    span = path_s[index + 1] - path_s[index]
    if span < 1e-6:
        return path_xy[index], float(path_yaw[index])
    mix = (station - path_s[index]) / span
    yaw = float(path_yaw[index])
    return path_xy[index] * (1.0 - mix) + path_xy[index + 1] * mix, yaw


def _station_lateral(points, path_xy, path_s, path_yaw):
    nearest = np.linalg.norm(points[:, None, :] - path_xy[None, :, :], axis=2).argmin(axis=1)
    yaw = path_yaw[nearest]
    tangent = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    left = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
    delta = points - path_xy[nearest]
    station = path_s[nearest] + np.sum(delta * tangent, axis=1)
    lateral = np.sum(delta * left, axis=1)
    return station, lateral


def _subsample(points, count):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= count:
        return points
    index = np.linspace(0, len(points) - 1, count).astype(np.int32)
    return points[index]


def _segment_distance(points, polyline):
    """Perpendicular distance from each point to a polyline."""
    if len(polyline) == 0:
        return np.full(len(points), 1e6)
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)
    start = polyline[:-1]
    step = polyline[1:] - start
    length2 = np.maximum(np.sum(step * step, axis=1), 1e-6)
    offset = points[:, None, :] - start[None, :, :]
    mix = np.clip(np.sum(offset * step[None, :, :], axis=2) / length2, 0.0, 1.0)
    closest = start[None, :, :] + mix[:, :, None] * step[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def _projected_segments(world, pose, project_many):
    ego = world_to_ego(np.asarray(world, dtype=np.float64), pose)
    pixels = project_many(ego)
    segments = []
    current = []
    for pixel in pixels:
        if not np.isfinite(pixel[0]):
            if len(current) >= 2:
                segments.append(np.asarray(current, dtype=np.float64))
            current = []
            continue
        current.append(pixel)
    if len(current) >= 2:
        segments.append(np.asarray(current, dtype=np.float64))
    return segments


def _pixel_distance(pixels, world, pose, project_many):
    segments = _projected_segments(world, pose, project_many)
    if not segments:
        return np.full(len(pixels), 64.0)
    return np.min([_segment_distance(pixels, segment) for segment in segments], axis=0)


def _curve_world(control_s, lateral, path_xy, path_s, path_yaw, step):
    samples = np.arange(control_s[0], control_s[-1] + step * 0.5, step)
    if len(samples) < 2:
        samples = np.array([control_s[0], control_s[-1]])
    offset = np.interp(samples, control_s, lateral)
    world = []
    for station, lat in zip(samples, offset):
        origin, yaw = _pose_at(float(station), path_xy, path_s, path_yaw)
        world.append(origin + lat * np.array([-np.sin(yaw), np.cos(yaw)]))
    return np.asarray(world, dtype=np.float64)


def _fit_controls(control_s, lateral, observations, poses, path_xy, path_s, path_yaw, project_many):
    """Move the lateral of each control so the projected curve hits the pixels."""
    values = np.asarray(lateral, dtype=np.float64).copy()
    count = len(values)

    def residual(current):
        world = _curve_world(control_s, current, path_xy, path_s, path_yaw, 1.0)
        parts = [
            _pixel_distance(pixels, world, poses[index], project_many)
            for index, pixels in observations
        ]
        if count >= 3:
            second = current[:-2] - 2.0 * current[1:-1] + current[2:]
            parts.append(second / 0.8)
        return np.concatenate(parts)

    def weights(error):
        smooth = max(count - 2, 0)
        pixel = error[:-smooth] if smooth else error
        scale = np.ones(len(error))
        large = np.abs(pixel) > 4.0
        scaled = np.ones(len(pixel))
        scaled[large] = 4.0 / np.maximum(np.abs(pixel[large]), 1e-6)
        scale[: len(pixel)] = scaled
        return scale

    for _ in range(6):
        base = residual(values)
        weight = weights(base)
        jacobian = np.zeros((len(base), count))
        for index in range(count):
            bumped = values.copy()
            bumped[index] += 0.05
            jacobian[:, index] = (residual(bumped) - base) / 0.05
        scale = np.sqrt(weight)
        system = (jacobian * scale[:, None]).T @ (jacobian * scale[:, None])
        target = (jacobian * scale[:, None]).T @ (base * scale)
        damp = 1e-3 * (np.trace(system) / max(count, 1) + 1e-6)
        try:
            step = np.linalg.solve(system + damp * np.eye(count), -target)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -0.4, 0.4)
        if float(np.linalg.norm(step)) < 1e-3:
            break
        candidate = values + step
        if np.sum(residual(candidate) ** 2 * weight) > np.sum(base ** 2 * weight):
            candidate = values + 0.5 * step
        values = candidate
    world = _curve_world(control_s, values, path_xy, path_s, path_yaw, 1.0)
    return values, world


def _controls_from_stats(stats):
    """One lateral per station bin. A sideways jump starts a new piece."""
    order = np.argsort(stats[:, 0])
    stats = stats[order]
    if stats[-1, 0] - stats[0, 0] < _MIN_SPAN_M:
        return []
    step = 8.0 if stats[-1, 0] - stats[0, 0] > 80.0 else 5.0
    edges = np.arange(stats[0, 0], stats[-1, 0] + step, step)
    bins = []
    for station in edges:
        window = stats[np.abs(stats[:, 0] - station) <= step]
        if len(window) == 0:
            continue
        bins.append([float(station), float(np.median(window[:, 1]))])
    if len(bins) < 2:
        return []
    bins = np.asarray(bins)
    pieces = []
    start = 0
    for index in range(1, len(bins)):
        jumped = abs(bins[index, 1] - bins[index - 1, 1]) > 2.5
        near = abs(bins[index, 0] - bins[index - 1, 0]) < 2.0 * step + 1.0
        if jumped and near:
            pieces.append(bins[start:index])
            start = index
    pieces.append(bins[start:])
    built = []
    for piece in pieces:
        if len(piece) < 2 or piece[-1, 0] - piece[0, 0] < _MIN_SPAN_M:
            continue
        built.append((piece[:, 0], piece[:, 1], piece))
    return built


def _fit_group(group, poses, path_xy, path_s, path_yaw, project_many, frames):
    fitted = []
    for control_s, lateral, piece in _controls_from_stats(group["stats"]):
        lo, hi = float(piece[0, 0]), float(piece[-1, 0])
        members = [
            chain
            for chain in group["chains"]
            if not (chain["station"].max() < lo - 8.0 or chain["station"].min() > hi + 8.0)
        ]
        if len({chain["index"] for chain in members}) < _MIN_FRAMES:
            continue
        observations = [(chain["index"], chain["pixels"]) for chain in members]
        _values, world = _fit_controls(
            control_s, lateral, observations, poses, path_xy, path_s, path_yaw, project_many
        )
        support = []
        errors = []
        for chain in members:
            distance = _pixel_distance(chain["pixels"], world, poses[chain["index"]], project_many)
            if float(np.median(distance)) <= _PIXEL_GATE:
                support.append(chain["index"])
                errors.append(distance)
        frames_hit = len(set(support))
        if frames_hit < _MIN_FRAMES or not errors:
            continue
        error = float(np.median(np.concatenate(errors)))
        if error > 8.0:
            continue
        support_set = set(support)
        samples = [{"token": frames[index]["token"], "ego": []} for index in sorted(support_set)]
        fitted.append(
            {
                "name": "road_line",
                "hits": frames_hit,
                "world": np.round(world, 3).tolist(),
                "samples": samples,
                "error_px": round(error, 2),
                "_members": [chain for chain in members if chain["index"] in support_set],
            }
        )
    return fitted


def _median_pixel_distance(pixels, worlds, pose, project_many) -> float:
    if not worlds:
        return 1e6
    distances = np.min(
        [_pixel_distance(pixels, world, pose, project_many) for world in worlds],
        axis=0,
    )
    return float(np.median(distances))


def _stations_overlap(chain, stats) -> bool:
    return not (chain["station"].max() < stats[:, 0].min() or chain["station"].min() > stats[:, 0].max())


def _facing_lateral(chain, stats) -> float:
    """Lateral gap where two pieces meet, not the gap between their middles.

    A merge moves sideways along its length, so the middles can be a lane
    apart while the ends that face each other are the same paint.
    """
    if chain["station"].max() < stats[:, 0].min():
        chain_lat = float(chain["lateral"][int(np.argmax(chain["station"]))])
        other_lat = float(stats[int(np.argmin(stats[:, 0])), 1])
    elif chain["station"].min() > stats[:, 0].max():
        chain_lat = float(chain["lateral"][int(np.argmin(chain["station"]))])
        other_lat = float(stats[int(np.argmax(stats[:, 0])), 1])
    else:
        return abs(float(np.median(chain["lateral"])) - float(np.median(stats[:, 1])))
    return abs(chain_lat - other_lat)


def _same_marking(chain, group, worlds, poses, project_many) -> bool:
    """A chain joins a marking when it lands on that marking's pixels.

    Lifted metres are only a way to extend a line into a frame the curve does
    not reach yet. The extension looks at the ends that face each other, so a
    taper stays one marking. Two edges a metre apart still do not.
    """
    if worlds:
        distance = _median_pixel_distance(chain["pixels"], worlds, poses[chain["index"]], project_many)
        if distance <= _PIXEL_GATE:
            return True
        if _stations_overlap(chain, group["stats"]) and distance < 64.0:
            return False
    if not _stations_overlap(chain, group["stats"]):
        gap = min(
            abs(float(chain["station"].min()) - float(group["stats"][:, 0].max())),
            abs(float(group["stats"][:, 0].min()) - float(chain["station"].max())),
        )
        # A dashed line has a gap of about this size. A hole across empty
        # road is longer, and the bridge test covers that.
        return gap <= 12.0 and _facing_lateral(chain, group["stats"]) < _GROUP_M
    lateral_gap = abs(float(np.median(chain["lateral"])) - float(np.median(group["stats"][:, 1])))
    return not worlds and lateral_gap < _GROUP_M


def _grow_groups(chains, poses, path_xy, path_s, path_yaw, project_many, frames):
    pending = sorted(chains, key=lambda chain: (-len(chain["pixels"]), chain["index"]))
    used = set()
    groups = []
    for seed in pending:
        if id(seed) in used:
            continue
        group = {"chains": [seed], "stats": np.asarray(seed["stats"], dtype=np.float64)}
        used.add(id(seed))
        worlds = []
        grew = True
        while grew:
            grew = False
            for other in pending:
                if id(other) in used:
                    continue
                if not _same_marking(other, group, worlds, poses, project_many):
                    continue
                group["chains"].append(other)
                group["stats"] = np.concatenate([group["stats"], other["stats"]], axis=0)
                used.add(id(other))
                grew = True
            if not grew:
                break
            fitted = _fit_group(group, poses, path_xy, path_s, path_yaw, project_many, frames)
            worlds = [np.asarray(item["world"], dtype=np.float64) for item in fitted]
            if not worlds:
                continue
            confirmed = []
            for chain in group["chains"]:
                distance = _median_pixel_distance(
                    chain["pixels"], worlds, poses[chain["index"]], project_many
                )
                if distance <= _PIXEL_GATE:
                    confirmed.append(chain)
            if confirmed and len(confirmed) != len(group["chains"]):
                group["chains"] = confirmed
                group["stats"] = np.concatenate([chain["stats"] for chain in confirmed], axis=0)
                fitted = _fit_group(group, poses, path_xy, path_s, path_yaw, project_many, frames)
                worlds = [np.asarray(item["world"], dtype=np.float64) for item in fitted]
        groups.append(group)
    return groups


def _split_gaps(curve, path_xy, path_s, path_yaw):
    """Drop the part of a curve that crosses road with no paint under it."""
    members = curve.get("_members") or []
    world = np.asarray(curve["world"], dtype=np.float64)
    if not members or len(world) < 2:
        return [curve]
    observed = np.concatenate([chain["station"] for chain in members])
    station, _lateral = _station_lateral(world, path_xy, path_s, path_yaw)
    supported = np.array([float(np.min(np.abs(observed - value))) <= 4.0 for value in station])
    runs = []
    start = None
    for index, flag in enumerate(list(supported) + [False]):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            runs.append((start, index))
            start = None
    pieces = []
    for lo, hi in runs:
        piece = world[lo:hi]
        if len(piece) < 2:
            continue
        piece_s, _piece_l = _station_lateral(piece, path_xy, path_s, path_yaw)
        if float(piece_s.max() - piece_s.min()) < _MIN_SPAN_M:
            continue
        copied = dict(curve)
        copied["world"] = np.round(piece, 3).tolist()
        pieces.append(copied)
    return pieces


def _same_paint(left, right, poses, project_many) -> bool:
    members = left.get("_members") or []
    world = np.asarray(right["world"], dtype=np.float64)
    if not members:
        return False
    distances = [
        _pixel_distance(chain["pixels"], world, poses[chain["index"]], project_many) for chain in members
    ]
    return float(np.median(np.concatenate(distances))) <= _PIXEL_GATE


def _merge_copies(curves, poses, project_many):
    """Two curves on the same dashes are one marking."""
    keep = [True] * len(curves)
    for i in range(len(curves)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(curves)):
            if not keep[j]:
                continue
            if not (
                _same_paint(curves[i], curves[j], poses, project_many)
                or _same_paint(curves[j], curves[i], poses, project_many)
            ):
                continue
            left_key = (curves[i]["hits"], -curves[i]["error_px"])
            right_key = (curves[j]["hits"], -curves[j]["error_px"])
            drop = j if left_key >= right_key else i
            keep[drop] = False
            if drop == i:
                break
    return [curve for curve, flag in zip(curves, keep) if flag]


def _ordered_curve(curve, path_xy, path_s, path_yaw):
    world = np.asarray(curve["world"], dtype=np.float64)
    station, lateral = _station_lateral(world, path_xy, path_s, path_yaw)
    order = np.argsort(station)
    return world[order], station[order], lateral[order]


def _uncross(curves, path_xy, path_s, path_yaw):
    """Keep each marking on its own side of the other.

    A merge may move sideways. It may not pass through the line beside it.
    Where two curves swap sides, the tails are exchanged so each one stays
    on the side it had before the crossing.
    """
    if len(curves) < 2:
        return curves
    ordered = [_ordered_curve(curve, path_xy, path_s, path_yaw) for curve in curves]
    for _ in range(6):
        changed = False
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                world_a, station_a, lateral_a = ordered[i]
                world_b, station_b, lateral_b = ordered[j]
                lo = max(float(station_a[0]), float(station_b[0]))
                hi = min(float(station_a[-1]), float(station_b[-1]))
                if hi - lo < 4.0:
                    continue
                grid = np.linspace(lo, hi, 24)
                diff = np.interp(grid, station_a, lateral_a) - np.interp(grid, station_b, lateral_b)
                sign = np.sign(diff)
                for index in range(1, len(sign)):
                    if sign[index] == 0:
                        sign[index] = sign[index - 1]
                if sign[0] == 0 or not np.any(np.diff(sign)):
                    continue
                flip = int(np.where(np.diff(sign))[0][0])
                if abs(float(diff[flip])) < 0.4 or abs(float(diff[flip + 1])) < 0.4:
                    continue
                cut = float(grid[flip] + grid[flip + 1]) / 2.0
                before_a = world_a[station_a < cut]
                after_a = world_a[station_a >= cut]
                before_b = world_b[station_b < cut]
                after_b = world_b[station_b >= cut]
                if min(len(before_a), len(after_a), len(before_b), len(after_b)) < 2:
                    continue
                # The other curve's tail is on this curve's old side.
                ordered[i] = _ordered_curve(
                    {"world": np.vstack([before_a, after_b])}, path_xy, path_s, path_yaw
                )
                ordered[j] = _ordered_curve(
                    {"world": np.vstack([before_b, after_a])}, path_xy, path_s, path_yaw
                )
                changed = True
        if not changed:
            break
    result = []
    for (world, _station, _lateral), curve in zip(ordered, curves):
        copied = dict(curve)
        copied["world"] = np.round(world, 3).tolist()
        result.append(copied)
    return result


def _select_ego(curves, path_xy, path_s, path_yaw):
    """Drop a marking that never comes near the road the car is driving.

    The nearest line on each side is not enough: the other lane, and a merge,
    are real markings. A line that stays more than a couple of lanes out
    (a sidewalk, the far curb) is the one that goes.
    """
    kept = []
    for curve in curves:
        world = np.asarray(curve["world"], dtype=np.float64)
        _station, lateral = _station_lateral(world, path_xy, path_s, path_yaw)
        if float(np.min(np.abs(lateral))) < 8.0:
            kept.append(curve)
    return kept


def fit_lane_curves(frames, project_many, lift) -> list[dict]:
    """One ground curve per marking, fitted so its projection hits the pixels.

    A chain joins a curve only when its pixels land on that curve. A stretch
    with no paint under it is removed, and two curves that cover the same
    dashes are one marking. A marking that never comes within two lanes of
    the car is not drawn. The neighbour lane and a merge are kept.
    """
    if not frames:
        return []
    poses = [frame["pose"] for frame in frames]
    path_xy, path_s, path_yaw = _path_from_poses(poses)
    chains = []
    for index, frame in enumerate(frames):
        for poly in frame.get("maps", {}).get("road_line", []):
            pixels = _subsample(np.asarray(poly, dtype=np.float64).reshape(-1, 2), 30)
            if len(pixels) < 4:
                continue
            ego = []
            for u, v in pixels:
                point = lift(float(u), float(v))
                if point is not None:
                    ego.append(point[:2])
            if len(ego) < 4:
                continue
            world = ego_to_world(np.asarray(ego, dtype=np.float64), poses[index])
            station, lateral = _station_lateral(world, path_xy, path_s, path_yaw)
            chains.append(
                {
                    "index": index,
                    "pixels": pixels,
                    "station": station,
                    "lateral": lateral,
                    "stats": np.stack([station, lateral], axis=1),
                }
            )
    if not chains:
        return []
    curves = []
    for group in _grow_groups(chains, poses, path_xy, path_s, path_yaw, project_many, frames):
        if len({chain["index"] for chain in group["chains"]}) < _MIN_FRAMES:
            continue
        curves.extend(_fit_group(group, poses, path_xy, path_s, path_yaw, project_many, frames))
    split = []
    for curve in curves:
        split.extend(_split_gaps(curve, path_xy, path_s, path_yaw))
    merged = _uncross(_merge_copies(split, poses, project_many), path_xy, path_s, path_yaw)
    selected = _select_ego(merged, path_xy, path_s, path_yaw)
    published = []
    for curve in selected:
        curve.pop("_members", None)
        curve["id"] = len(published) + 1
        published.append(curve)
    return published


def merge_object_tracks(tracks: list[dict], token_time: dict[str, float]) -> list[dict]:
    """Join tracks that are the same object seen twice.

    Two tracks merge when they occupy the same place at the same time
    (median under 2.2 m on at least two shared frames), or when one ends
    and the other starts within 0.4 s and 2.5 m. Adjacent-lane cars stay
    apart: a lane is about 3.5 m wide.
    """
    parent = list(range(len(tracks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def link(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    series = []
    for track in tracks:
        points = []
        for sample in track["samples"]:
            t = token_time.get(sample["token"])
            if t is None:
                continue
            points.append((float(t), sample))
        points.sort(key=lambda item: item[0])
        series.append(points)

    for i, left in enumerate(tracks):
        if not series[i]:
            continue
        left_by = {sample["token"]: (t, sample) for t, sample in series[i]}
        for j in range(i + 1, len(tracks)):
            right = tracks[j]
            if left["name"] != right["name"] or not series[j]:
                continue
            y_left = float(np.median([sample["world"][1] for _t, sample in series[i]]))
            y_right = float(np.median([sample["world"][1] for _t, sample in series[j]]))
            # A lane is about 3.5 m. Tracks in different lanes are not the same car,
            # even when one bad frame puts a box on top of the other.
            if abs(y_left - y_right) > 2.0:
                continue
            distances = []
            for t, sample in series[j]:
                other = left_by.get(sample["token"])
                if other is None:
                    continue
                a = other[1]["world"]
                b = sample["world"]
                distances.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
            if len(distances) >= 2 and float(np.median(distances)) <= 2.2:
                link(i, j)
                continue
            end_i, start_j = series[i][-1], series[j][0]
            end_j, start_i = series[j][-1], series[i][0]
            joins = (
                (start_j[0] - end_i[0], end_i[1]["world"], start_j[1]["world"]),
                (start_i[0] - end_j[0], end_j[1]["world"], start_i[1]["world"]),
            )
            for gap, a, b in joins:
                if 0 <= gap <= 0.4 and float(np.hypot(a[0] - b[0], a[1] - b[1])) <= 2.5:
                    link(i, j)
                    break

    groups: dict[int, list[int]] = {}
    for i in range(len(tracks)):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for indexes in groups.values():
        by_token: dict[str, tuple[float, dict]] = {}
        names = tracks[indexes[0]]
        for index in indexes:
            for t, sample in series[index]:
                current = by_token.get(sample["token"])
                if current is None or sample["score"] > current[1]["score"]:
                    by_token[sample["token"]] = (t, sample)
        samples = [sample for _t, sample in sorted(by_token.values(), key=lambda item: item[0])]
        if samples:
            centre_y = float(np.median([sample["world"][1] for sample in samples]))
            samples = [sample for sample in samples if abs(sample["world"][1] - centre_y) <= 3.0]
        if not samples:
            continue
        merged.append(
            {
                "id": len(merged) + 1,
                "name": names["name"],
                "label": names["label"],
                "hits": len(samples),
                "samples": samples,
            }
        )
    merged.sort(key=lambda track: (track["name"], track["samples"][0]["world"][0] if track["samples"] else 0))
    for index, track in enumerate(merged, start=1):
        track["id"] = index
    return merged
