"""Byte-identity tests for the fast _fit_controls residual in qdtrack/tracks.py.

    python qdtrack/test_fast_lanes.py                 # 8,000 randomized cases
    python qdtrack/test_fast_lanes.py --cases 200     # quick run
Also collectable by pytest (test_random_cases, test_degenerate_cases, test_switch_off).
Every case compares _fast_pixel_distances with the per-observation _pixel_distance loop it replaces,
byte for byte (dtype, shape, bits). Exit code 1 on any mismatch.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdtrack import tracks as T  # noqa: E402


class _Camera:
    """CylinderCamera.project_many of scripts/overlay.py with the FC1 Wide intrinsics (no cv2/torch needed)."""

    def __init__(self, fx=591.5789, fy=591.5789, cx=576.0, cy=480.0, position=(1.97, 0.274, 1.544)):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.position = np.asarray(position, dtype=np.float64)

    def project_many(self, xy):
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


def _reference(observations, poses, world, project_many):
    parts = [T._pixel_distance(p, world, poses[i], project_many) for i, p in observations]
    return np.concatenate(parts) if parts else np.zeros(0)


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _random_case(rng, trial):
    O = int(rng.integers(0, 6))
    N = int(rng.integers(2, 140))
    s = np.linspace(0, rng.uniform(5, 150), N)
    world = np.stack([s * np.cos(rng.uniform(-0.3, 0.3)) + rng.normal(0, 0.2, N),
                      rng.uniform(-8, 8) + rng.normal(0, 0.5, N)], 1)
    poses, observations = {}, []
    for o in range(O):
        poses[o] = {"x": float(rng.uniform(-20, 60)), "y": float(rng.normal(0, 3)), "yaw": float(rng.normal(0, 0.3))}
        n = int(rng.integers(0, 30))
        p = np.stack([rng.uniform(-100, 1300, n), rng.uniform(300, 700, n)], 1)
        if trial % 4 == 0:
            p = np.round(p).astype(np.int64)
        elif trial % 4 == 1:
            p = p.astype(np.float32)
        observations.append((o, p))
    return observations, poses, world


def run_random(cases=8000, seed=1):
    cam, rng = _Camera(), np.random.default_rng(seed)
    bad = 0
    for trial in range(cases):
        observations, poses, world = _random_case(rng, trial)
        ref = _reference(observations, poses, world, cam.project_many)
        got = T._fast_pixel_distances(T._fast_prepare(observations, poses), world, cam.project_many)
        bad += not _same(ref, got)
    return bad


def test_random_cases():
    assert run_random(int(os.environ.get("FBT_FAST_TEST_CASES", "8000"))) == 0


def test_degenerate_cases():
    cam = _Camera()
    pose = {0: {"x": 0.0, "y": 0.0, "yaw": 0.0}}
    px = np.array([[500.0, 600.0], [700.0, 550.0]])
    worlds = [np.array([[10.0, 0.0], [20.0, 1.0]]),        # one segment
              np.array([[-5.0, 0.0], [-1.0, 0.0]]),        # behind the camera: no segment -> 64.0
              np.array([[-5.0, 0.0], [10.0, 0.0], [-1.0, 0.0], [12.0, 1.0]])]  # isolated finite points
    for world in worlds:
        for obs in ([(0, px)], [(0, px[:0])], []):
            ref = _reference(obs, pose, world, cam.project_many)
            got = T._fast_pixel_distances(T._fast_prepare(obs, pose), world, cam.project_many)
            assert _same(ref, got), (world.tolist(), len(obs))


def test_switch_off():
    import importlib
    os.environ["FBT_FAST_LANES"] = "0"
    try:
        assert importlib.reload(T)._FAST_LANES is False
    finally:
        os.environ.pop("FBT_FAST_LANES")
        importlib.reload(T)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    test_degenerate_cases()
    test_switch_off()
    bad = run_random(a.cases, a.seed)
    print("cases", a.cases, "mismatches", bad, "numpy", np.__version__)
    sys.exit(1 if bad else 0)
