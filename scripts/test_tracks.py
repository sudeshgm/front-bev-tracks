#!/usr/bin/env python3
"""CPU checks for calibration, ego integration, and both trackers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdtrack.geometry import build_lidar2img, sensor_to_ego  # noqa: E402
from qdtrack.tracks import ego_to_world, integrate_ego, polylines_from_map, track_lanes, track_objects, world_to_ego  # noqa: E402


def test_front_camera_sees_the_road():
    cfg = json.loads(Path("/tmp/a2d2/cams_lidars.json").read_text())
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


def test_parallel_copies_collapse():
    frames = []
    for i in range(4):
        pose = {"x": i * 8.0, "y": 0.0, "yaw": 0.0}
        # Two traces of the same edge, 0.4 m apart, each covering the next stretch.
        ego = np.array([[x - pose["x"], -1.5 + (0.4 if i % 2 else 0.0)] for x in range(i * 8, i * 8 + 12)], dtype=np.float64)
        frames.append({"token": str(i), "t": i * 0.2, "pose": pose, "maps": {"road_line": [], "road_edge": [ego]}})
    tracks = track_lanes(frames, gate=0.2)
    assert len(tracks) == 1, [(t["id"], t["hits"]) for t in tracks]
    xs = [p[0] for p in tracks[0]["world"]]
    assert max(xs) - min(xs) > 25


def test_map_raster_becomes_a_line():
    raster = np.zeros((400, 200), dtype=np.uint8)
    # road_line is class 2. Draw a vertical strip (constant column = constant y).
    raster[40:360, 120] = 2
    raster[40:360, 121] = 2
    lines = polylines_from_map(raster, "road_line")
    assert len(lines) == 1, len(lines)
    span = lines[0][:, 0].max() - lines[0][:, 0].min()
    assert span > 20


if __name__ == "__main__":
    test_front_camera_sees_the_road()
    test_ego_roundtrip_and_integration()
    test_object_ids_survive_a_gap()
    test_lane_association()
    test_map_raster_becomes_a_line()
    print("ok")
