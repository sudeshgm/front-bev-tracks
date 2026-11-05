#!/usr/bin/env python3
"""CPU checks for calibration, ego integration, and both trackers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdtrack.geometry import build_lidar2img, pixel_to_ego, sensor_to_ego  # noqa: E402
from qdtrack.tracks import (  # noqa: E402
    ego_to_world,
    fit_lane_curves,
    integrate_ego,
    merge_object_tracks,
    polylines_from_map,
    track_lanes,
    track_objects,
    world_to_ego,
    _uncross,
)


def test_front_camera_sees_the_road():
    path = Path("/tmp/a2d2/cams_lidars.json")
    if not path.exists():
        print("skip front-camera projection, no local calibration file")
        return
    cfg = json.loads(path.read_text())
    camera = cfg["cameras"]["front_center"]
    rotation, translation = sensor_to_ego(camera["view"])
    intrinsic = np.asarray(camera["CamMatrix"], dtype=np.float64)
    matrix = build_lidar2img(intrinsic, rotation, translation)
    # A point 20 m ahead on the ground should land near the principal point,
    # slightly below it, and in front of the camera.
    point = np.array([translation[0] + 20.0, translation[1], 0.0, 1.0])
    projected = matrix @ point
    assert projected[2] > 0
    u, v = projected[0] / projected[2], projected[1] / projected[2]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    assert abs(u - cx) < 80, (u, cx)
    assert v > cy - 5, (v, cy)


def test_ego_roundtrip_and_integration():
    samples = [
        {"t": 0.0, "speed_mps": 10.0, "yaw_rate": 0.0},
        {"t": 1.0, "speed_mps": 10.0, "yaw_rate": 0.0},
        {"t": 2.0, "speed_mps": 10.0, "yaw_rate": 0.0},
    ]
    poses = integrate_ego(samples)
    assert abs(poses[-1]["x"] - 20.0) < 1e-6
    assert abs(poses[-1]["y"]) < 1e-6
    pose = {"x": 5.0, "y": -2.0, "yaw": 0.3}
    points = np.array([[1.0, 2.0], [8.0, -1.0]])
    back = world_to_ego(ego_to_world(points, pose), pose)
    assert np.allclose(back, points)


def test_object_ids_survive_a_gap():
    def frame(token, t, pose, centres):
        boxes, scores, labels = [], [], []
        for x, y in centres:
            boxes.append([x, y, 0.0, 1.8, 4.2, 1.5, 0.0, 2.0, 0.0])
            scores.append(0.8)
            labels.append(0)
        return {
            "token": token,
            "t": t,
            "pose": pose,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

    frames = [
        frame("a", 0.0, {"x": 0, "y": 0, "yaw": 0}, [(10, 0), (12, 3)]),
        frame("b", 0.2, {"x": 2, "y": 0, "yaw": 0}, [(8, 0), (10, 3)]),
        frame("c", 0.4, {"x": 4, "y": 0, "yaw": 0}, [(8, 3)]),  # first car missed
        frame("d", 0.6, {"x": 6, "y": 0, "yaw": 0}, [(4, 0), (6, 3)]),
    ]
    tracks = track_objects(frames, score_min=0.2)
    by_name = {}
    for track in tracks:
        start = tuple(track["samples"][0]["world"])
        by_name[start] = [s["token"] for s in track["samples"]]
    assert len(tracks) == 2, tracks
    # The car that dropped out for one frame is still one track.
    long = max(tracks, key=lambda t: t["hits"])
    assert long["hits"] >= 3


def test_lane_association():
    line = np.array([[i, 1.5] for i in range(0, 20)], dtype=np.float64)
    frames = []
    for i, token in enumerate("abcd"):
        pose = {"x": i * 2.0, "y": 0.0, "yaw": 0.0}
        # The same world line, expressed in the moving ego frame.
        ego = line.copy()
        ego[:, 0] -= pose["x"]
        frames.append({"token": token, "t": i * 0.2, "pose": pose, "maps": {"road_line": [ego], "road_edge": []}})
    tracks = track_lanes(frames, gate=1.6)
    assert len(tracks) == 1, tracks
    assert tracks[0]["hits"] == 4


def test_reprojection_curve_keeps_gap_bend_and_outlier():
    import importlib.util

    spec = importlib.util.spec_from_file_location("overlay", ROOT / "scripts" / "overlay.py")
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    camera = overlay.CylinderCamera(600.0, 600.0, 320.0, 240.0, [1.5, 0.0, 1.4], 640, 480)
    samples = [{"t": i * 0.2, "speed_mps": 8.0, "yaw_rate": 0.0} for i in range(20)]
    poses = integrate_ego(samples)

    def lateral(name, station):
        if name == "left":
            return 1.75
        # A real bend: the right marking moves out as station grows.
        if station < 12.0:
            return -1.75
        return -1.75 - 0.004 * (station - 12.0) ** 2

    def chain(name, pose, corrupt):
        world = []
        for station in np.arange(pose["x"] + 6.0, pose["x"] + 28.0, 1.0):
            lat = lateral(name, station) + (0.9 if corrupt else 0.0)
            world.append([station, lat])
        pixels = camera.project_many(world_to_ego(np.asarray(world), pose))
        keep = np.isfinite(pixels[:, 0])
        keep &= (pixels[:, 0] > 0) & (pixels[:, 0] < camera.width)
        keep &= (pixels[:, 1] > camera.cy + 2) & (pixels[:, 1] < camera.height)
        pixels = pixels[keep]
        if len(pixels) < 6:
            return None
        pixels = pixels + np.array([0.3, -0.2])
        return pixels

    frames = []
    for index, pose in enumerate(poses):
        lines = []
        for name in ("left", "right"):
            # Eight frames of the right marking are a curb, 0.9 m outside the paint.
            corrupt = name == "right" and index >= 12
            pixels = chain(name, pose, corrupt)
            if pixels is not None:
                lines.append(pixels)
        frames.append(
            {
                "token": str(index),
                "t": samples[index]["t"],
                "pose": pose,
                "maps": {"road_line": lines, "road_edge": []},
            }
        )
    curves = fit_lane_curves(frames, camera.project_many, camera.lift)
    assert len(curves) == 2, [(c["id"], c["hits"], c["error_px"]) for c in curves]
    path_xy = np.array([[p["x"], p["y"]] for p in poses])
    for curve in curves:
        world = np.asarray(curve["world"], dtype=np.float64)
        nearest = np.linalg.norm(world[:, None, :] - path_xy[None, :, :], axis=2).argmin(1)
        # Path is straight along x, yaw 0, so lateral is y.
        lateral_y = world[:, 1]
        station = world[:, 0]
        early = lateral_y[station < 12.0]
        late = lateral_y[station > 24.0]
        assert len(early) and len(late)
        if float(np.median(early)) > 0:
            assert abs(float(np.median(early)) - 1.75) < 0.25, float(np.median(early))
        else:
            # The curb frames must not pull the paint, and the bend must remain.
            assert abs(float(np.median(early)) + 1.75) < 0.25, float(np.median(early))
            assert float(np.median(late)) < -2.4, (float(np.median(early)), float(np.median(late)))
        assert curve["error_px"] < 4.0, curve["error_px"]
        del nearest


def test_image_lane_chain_is_pixels():
    import importlib.util

    spec = importlib.util.spec_from_file_location("overlay", ROOT / "scripts" / "overlay.py")
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    camera = overlay.CylinderCamera(600.0, 600.0, 320.0, 240.0, [1.5, 0.0, 1.4], 640, 480)
    mask = np.zeros((480, 640), dtype=bool)
    mask[260:400, 300:303] = True
    mask[260:400, 360:364] = True
    mask[260:400, 10:80] = True  # too wide to be a marking
    lines = overlay.lane_polylines(mask, camera)
    assert len(lines) == 2, len(lines)
    for line in lines:
        assert line[:, 1].min() > camera.cy
        assert 290 < line[:, 0].mean() < 380


def test_map_raster_becomes_a_line():
    raster = np.zeros((400, 200), dtype=np.uint8)
    # road_line is class 2. Draw a vertical strip (constant column = constant y).
    raster[40:360, 120] = 2
    raster[40:360, 121] = 2
    lines = polylines_from_map(raster, "road_line")
    assert len(lines) == 1, len(lines)
    span = lines[0][:, 0].max() - lines[0][:, 0].min()
    assert span > 20


def test_split_vehicle_ids_join():
    def track(tid, samples):
        return {"id": tid, "name": "vehicle", "label": 0, "hits": len(samples), "samples": samples}

    def sample(token, x, y, score=0.9):
        return {"token": token, "score": score, "box": [0, 0, 0, 4, 2, 1.5, 0, 0, 0], "world": [x, y]}

    same_a = track(1, [sample("a", 10, 7), sample("b", 12, 7), sample("c", 14, 7.2)])
    same_b = track(2, [sample("b", 12.4, 7.4), sample("c", 14.3, 7.1), sample("d", 16, 7)])
    other = track(3, [sample("b", 12, 11), sample("c", 14, 11), sample("d", 16, 11)])
    times = {"a": 0.0, "b": 0.2, "c": 0.4, "d": 0.6}
    merged = merge_object_tracks([same_a, same_b, other], times)
    assert len(merged) == 2, [(t["id"], t["hits"]) for t in merged]
    hits = sorted(t["hits"] for t in merged)
    assert hits == [3, 4]


def test_pixel_lifts_through_the_camera_centre():
    intrinsic = np.array([[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1.0]])
    ego = pixel_to_ego(50, 40, 10.0, intrinsic, np.eye(3), np.zeros(3))
    assert np.allclose(ego, [0, 0, 10], atol=1e-6)


def test_cylinder_lift_roundtrip():
    import importlib.util

    spec = importlib.util.spec_from_file_location("overlay", ROOT / "scripts" / "overlay.py")
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    camera = overlay.CylinderCamera(700.0, 700.0, 640.0, 360.0, [2.0, 0.0, 1.5], 1280, 720)
    pixel = camera.project(22.0, -1.5, 0.0)
    point = camera.lift(*pixel)
    assert point is not None
    assert abs(point[0] - 22.0) < 1e-6 and abs(point[1] + 1.5) < 1e-6


def _camera_and_drive(seconds=16):
    import importlib.util

    spec = importlib.util.spec_from_file_location("overlay", ROOT / "scripts" / "overlay.py")
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    camera = overlay.CylinderCamera(600.0, 600.0, 320.0, 240.0, [1.5, 0.0, 1.4], 640, 480)
    samples = [{"t": i * 0.2, "speed_mps": 8.0, "yaw_rate": 0.0} for i in range(seconds)]
    return overlay, camera, integrate_ego(samples), samples


def _pixels_of(camera, pose, lateral_at):
    world = []
    for station in np.arange(pose["x"] + 6.0, pose["x"] + 28.0, 1.0):
        world.append([station, lateral_at(station)])
    pixels = camera.project_many(world_to_ego(np.asarray(world), pose))
    keep = np.isfinite(pixels[:, 0])
    keep &= (pixels[:, 0] > 0) & (pixels[:, 0] < camera.width)
    keep &= (pixels[:, 1] > camera.cy + 2) & (pixels[:, 1] < camera.height)
    pixels = pixels[keep]
    if len(pixels) < 6:
        return None
    return pixels


def test_copies_of_one_marking_are_one_curve():
    _overlay, camera, poses, samples = _camera_and_drive()
    frames = []
    for index, pose in enumerate(poses):
        first = _pixels_of(camera, pose, lambda station: 1.75)
        second = None if first is None else first + np.array([3.0, -1.0])
        lines = [line for line in (first, second) if line is not None]
        frames.append(
            {"token": str(index), "t": samples[index]["t"], "pose": pose, "maps": {"road_line": lines, "road_edge": []}}
        )
    curves = fit_lane_curves(frames, camera.project_many, camera.lift)
    assert len(curves) == 1, [(c["id"], c["hits"]) for c in curves]


def test_a_gap_in_the_paint_is_not_bridged():
    _overlay, camera, poses, samples = _camera_and_drive()
    frames = []
    for index, pose in enumerate(poses):
        lines = []
        for lo, hi in ((pose["x"] + 4.0, pose["x"] + 16.0), (pose["x"] + 55.0, pose["x"] + 70.0)):
            world = [[station, 1.75] for station in np.arange(lo, hi, 1.0)]
            pixels = camera.project_many(world_to_ego(np.asarray(world), pose))
            keep = np.isfinite(pixels[:, 0])
            keep &= (pixels[:, 0] > 0) & (pixels[:, 0] < camera.width)
            keep &= (pixels[:, 1] > camera.cy + 2) & (pixels[:, 1] < camera.height)
            pixels = pixels[keep]
            if len(pixels) >= 6:
                lines.append(pixels)
        frames.append(
            {"token": str(index), "t": samples[index]["t"], "pose": pose, "maps": {"road_line": lines, "road_edge": []}}
        )
    curves = fit_lane_curves(frames, camera.project_many, camera.lift)
    for curve in curves:
        station = np.asarray(curve["world"], dtype=np.float64)[:, 0]
        assert not ((station < 30.0).any() and (station > 55.0).any()), (float(station.min()), float(station.max()))


def test_a_merge_taper_stays_one_marking():
    _overlay, camera, poses, samples = _camera_and_drive()
    frames = []
    for index, pose in enumerate(poses):
        lines = []
        near_end = pose["x"] + 14.0
        far_start = pose["x"] + 18.0

        def opening(station, far_start=far_start):
            return 1.9 + 0.18 * (station - far_start)

        for lo, hi, lateral_at in (
            (pose["x"] + 4.0, near_end, lambda station: 1.75),
            (far_start, pose["x"] + 36.0, opening),
        ):
            world = [[station, lateral_at(station)] for station in np.arange(lo, hi, 1.0)]
            pixels = camera.project_many(world_to_ego(np.asarray(world, dtype=np.float64), pose))
            keep = np.isfinite(pixels[:, 0])
            keep &= (pixels[:, 0] > 0) & (pixels[:, 0] < camera.width)
            keep &= (pixels[:, 1] > camera.cy + 2) & (pixels[:, 1] < camera.height)
            pixels = pixels[keep]
            if len(pixels) >= 6:
                lines.append(pixels)
        frames.append(
            {"token": str(index), "t": samples[index]["t"], "pose": pose, "maps": {"road_line": lines, "road_edge": []}}
        )
    curves = fit_lane_curves(frames, camera.project_many, camera.lift)
    spanned = []
    for curve in curves:
        lateral = np.asarray(curve["world"], dtype=np.float64)[:, 1]
        spanned.append((round(float(lateral.min()), 2), round(float(lateral.max()), 2)))
    assert any(lo < 2.2 and hi > 4.0 for lo, hi in spanned), spanned


def test_two_markings_do_not_cross():
    path_xy = np.array([[float(i), 0.0] for i in range(40)])
    path_s = path_xy[:, 0].copy()
    path_yaw = np.zeros(len(path_xy))
    left_then_right = [[float(x), 1.5 if x < 20 else -1.5] for x in range(5, 35)]
    right_then_left = [[float(x), -1.5 if x < 20 else 1.5] for x in range(5, 35)]
    curves = [
        {"world": left_then_right, "hits": 10, "error_px": 1.0, "name": "road_line"},
        {"world": right_then_left, "hits": 10, "error_px": 1.0, "name": "road_line"},
    ]
    result = _uncross(curves, path_xy, path_s, path_yaw)
    for curve in result:
        lateral = np.asarray(curve["world"], dtype=np.float64)[:, 1]
        assert lateral.min() > 0.5 or lateral.max() < -0.5, lateral


def test_a_far_line_is_not_the_ego_lane():
    _overlay, camera, poses, samples = _camera_and_drive()
    frames = []
    for index, pose in enumerate(poses):
        lines = []
        for lateral in (1.75, 5.0, 14.0):
            pixels = _pixels_of(camera, pose, lambda station, lateral=lateral: lateral)
            if pixels is not None:
                lines.append(pixels)
        frames.append(
            {"token": str(index), "t": samples[index]["t"], "pose": pose, "maps": {"road_line": lines, "road_edge": []}}
        )
    curves = fit_lane_curves(frames, camera.project_many, camera.lift)
    laterals = sorted(round(float(np.median(np.asarray(curve["world"])[:, 1])), 1) for curve in curves)
    assert laterals == [1.8, 5.0] or all(abs(v - 1.75) < 0.4 or abs(v - 5.0) < 0.4 for v in laterals), laterals
    assert len(curves) == 2, laterals


def test_a_gap_keeps_the_speed_and_the_track_continues():
    def frame(token, t, x, y, detections):
        boxes, scores, labels = [], [], []
        for dx, dy in detections:
            boxes.append([dx, dy, 0.0, 1.8, 4.2, 1.5, 0.0, 0.0, 0.0])
            scores.append(0.9)
            labels.append(0)
        return {
            "token": token,
            "t": t,
            "pose": {"x": x, "y": y, "yaw": 0.0},
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

    # Seen from frame 8. A 0.6 s gap in the same lane is still this car.
    seen = [(1.6, 20.0), (1.8, 18.0), (2.0, 16.0), (2.6, 10.0)]
    frames = []
    for index in range(16):
        t = round(index * 0.2, 3)
        detection = [(x, 0.0) for st, x in seen if abs(st - t) < 1e-6]
        frames.append(frame(str(index), t, 0.0, 0.0, detection))
    tracks = track_objects(frames, score_min=0.2, max_gap=15)
    assert len(tracks) == 1, [(track["id"], track["hits"]) for track in tracks]
    assert tracks[0]["hits"] == 4
    face = tracks[0]["samples"][0].get("face")
    assert face is not None
    centre = tracks[0]["samples"][0]["world"]
    assert abs(centre[0]) + abs(centre[1]) > abs(face[0]) + abs(face[1])


def test_a_sideways_jump_is_a_different_car():
    def frame(token, t, centres):
        boxes, scores, labels = [], [], []
        for x, y in centres:
            boxes.append([x, y, 0.0, 1.8, 4.2, 1.5, 0.0, 0.0, 0.0])
            scores.append(0.9)
            labels.append(0)
        return {
            "token": token,
            "t": t,
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

    frames = []
    for index in range(8):
        y = 3.6 if index < 5 else 9.6
        frames.append(frame(str(index), round(index * 0.067, 3), [(8.0, y)]))
    tracks = track_objects(frames, score_min=0.2, max_gap=15)
    first = min(tracks, key=lambda track: int(track["samples"][0]["token"]))
    followed = [sample["token"] for sample in first["samples"] if not sample.get("predicted")]
    assert followed == ["0", "1", "2", "3", "4"], followed
    assert len(tracks) == 2


def test_a_close_approach_is_not_left_behind():
    import importlib.util

    spec = importlib.util.spec_from_file_location("overlay", ROOT / "scripts" / "overlay.py")
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    poses = [{"x": index * 0.1, "y": 0.0, "yaw": 0.0} for index in range(40)]
    samples = []
    car_x = 50.0
    for index, pose in enumerate(poses):
        car_x -= 0.3 if index < 20 else 1.4
        samples.append(
            {
                "token": str(index),
                "score": 0.9,
                "box": [car_x - pose["x"], 2.0, 0, 1.8, 4.2, 1.5, 0, 0, 0],
                "world": [car_x, 2.0],
            }
        )
    last = samples[-1]["world"][0]
    track = {"id": 1, "name": "vehicle", "label": 0, "hits": len(samples), "samples": samples}
    overlay.smooth_world([track], poses, 0.067)
    end = track["samples"][-1]["world"][0]
    assert abs(end - last) < 2.5, (end, last)


def test_a_sideways_wander_does_not_turn_the_heading():
    import importlib.util

    spec = importlib.util.spec_from_file_location("overlay", ROOT / "scripts" / "overlay.py")
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    poses = [{"x": 0.0, "y": 0.0, "yaw": 0.0} for _ in range(30)]
    samples = []
    for index in range(30):
        along = 40.0 - index * 1.0
        side = 2.0 if index <= 18 else 2.0 - (index - 18) * 0.4
        samples.append(
            {
                "token": str(index),
                "score": 0.9,
                "box": [along, side, 0, 1.8, 4.2, 1.5, 0, 0, 0],
                "world": [along, side],
            }
        )
    last_along = samples[-1]["world"][0]
    track = {"id": 1, "name": "vehicle", "label": 0, "hits": len(samples), "samples": samples}
    overlay.smooth_world([track], poses, 0.067)
    worlds = np.array([sample["world"] for sample in track["samples"]], dtype=np.float64)
    delta = worlds[-1] - worlds[-12]
    assert abs(delta[0]) > 8.0, delta
    assert abs(delta[1]) < 1.5, delta
    assert abs(worlds[-1, 0] - last_along) < 2.5, (worlds[-1, 0], last_along)


def test_oncoming_range_jump_keeps_the_id():
    def frame(token, t, pose, centres):
        boxes, scores, labels = [], [], []
        for x, y in centres:
            boxes.append([x, y, 0.0, 1.8, 4.2, 1.5, 0.0, 0.0, 0.0])
            scores.append(0.9)
            labels.append(0)
        return {"token": token, "t": t, "pose": pose, "boxes": boxes, "scores": scores, "labels": labels}

    frames = [
        frame("a", 0.0, {"x": 0, "y": 0, "yaw": 0}, [(30, 0)]),
        frame("b", 0.1, {"x": 1, "y": 0, "yaw": 0}, [(21, 0), (30, 3.5)]),
    ]
    tracks = track_objects(frames, score_min=0.2)
    assert len(tracks) == 2, [(t["id"], t["hits"]) for t in tracks]
    kept = max(tracks, key=lambda t: t["hits"])
    assert kept["hits"] == 2


def test_oncoming_car_is_not_held_still():
    import importlib.util

    spec = importlib.util.spec_from_file_location("overlay", ROOT / "scripts" / "overlay.py")
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    samples = [{"t": i * 0.2, "speed_mps": 10.0, "yaw_rate": 0.0} for i in range(16)]
    poses = integrate_ego(samples)
    track_samples = []
    for index, pose in enumerate(poses):
        world_x = 50.0 - index * 2.0
        track_samples.append(
            {
                "token": str(index),
                "score": 0.9,
                "box": [world_x - pose["x"], 1.5, 0, 1.8, 4.2, 1.5, 0, 0, 0],
                "world": [world_x, 1.5],
            }
        )
    track = {"id": 1, "name": "vehicle", "label": 0, "hits": len(track_samples), "samples": track_samples}
    overlay.smooth_world([track], poses, 0.2)
    travelled = abs(track["samples"][-1]["world"][0] - track["samples"][0]["world"][0])
    assert travelled > 10.0, travelled


if __name__ == "__main__":
    test_front_camera_sees_the_road()
    test_ego_roundtrip_and_integration()
    test_object_ids_survive_a_gap()
    test_lane_association()
    test_reprojection_curve_keeps_gap_bend_and_outlier()
    test_image_lane_chain_is_pixels()
    test_map_raster_becomes_a_line()
    test_split_vehicle_ids_join()
    test_pixel_lifts_through_the_camera_centre()
    test_cylinder_lift_roundtrip()
    test_copies_of_one_marking_are_one_curve()
    test_a_gap_in_the_paint_is_not_bridged()
    test_a_far_line_is_not_the_ego_lane()
    test_a_merge_taper_stays_one_marking()
    test_two_markings_do_not_cross()
    test_oncoming_range_jump_keeps_the_id()
    test_a_gap_keeps_the_speed_and_the_track_continues()
    test_a_sideways_jump_is_a_different_car()
    test_a_close_approach_is_not_left_behind()
    test_a_sideways_wander_does_not_turn_the_heading()
    test_oncoming_car_is_not_held_still()
    print("ok")
