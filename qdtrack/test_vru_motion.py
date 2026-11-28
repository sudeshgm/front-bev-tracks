"""Tests for qdtrack/vru_motion.py (pedestrian/cyclist smoother and tag windows).

    python -m pytest qdtrack/test_vru_motion.py -q
    python qdtrack/test_vru_motion.py
Synthetic tracks with the same monocular noise model the smoother assumes:
along-sight noise ~ r^2, lateral ~ r. Seeds are fixed.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdtrack import vru_motion as V  # noqa: E402

HZ = 10.0
DURATION = 10.0


def _ego_path(times, speed=5.0):
    """Ego drives along +x."""
    return np.stack([speed * times, np.zeros_like(times)], axis=1)


def _observe(truth, egos, rng, config=V.DEFAULT_CONFIG):
    out = np.zeros_like(truth)
    for k in range(len(truth)):
        cov = V.measurement_covariance(truth[k], egos[k], config, correlated=False)
        out[k] = rng.multivariate_normal(truth[k], cov)
    return out


def _times():
    return np.arange(0.0, DURATION, 1.0 / HZ)


def test_standing_pedestrian_is_slow_and_stationary():
    times = _times()
    egos = _ego_path(times, speed=0.0)  # ego waiting at the junction
    truth = np.tile([20.0, 4.0], (len(times), 1))  # ~20 m away
    for seed in range(20):
        rng = np.random.default_rng(seed)
        raw = _observe(truth, egos, rng)
        raw_speed = np.median(np.linalg.norm(np.diff(raw, axis=0), axis=1) * HZ)
        m = V.smooth_vru(times, raw, egos, "person")
        assert raw_speed > 5.0, raw_speed  # the jitter the old pipeline saw as motion
        assert m.speed() < 0.8, (seed, m.speed())
        assert m.stationary(), (seed, m.stationary_report())


def test_standing_pedestrian_while_ego_drives_past():
    times = _times()
    egos = _ego_path(times, speed=3.0)  # ego creeping through a turn
    truth = np.tile([35.0, 6.0], (len(times), 1))  # 35 m -> 6 m: along-sight noise ~8 m -> 0.3 m
    for seed in range(20):
        m = V.smooth_vru(times, _observe(truth, egos, np.random.default_rng(100 + seed)), egos, "person")
        assert m.speed() < 1.0, (seed, m.speed())
        assert m.stationary(), (seed, m.stationary_report())


def test_standing_pedestrian_with_slow_pitch_drift():
    """Slow pitch error (braking into the turn) moves a standing person along
    the line of sight for seconds; that must not read as walking."""
    times = _times()
    egos = _ego_path(times, speed=0.0)
    truth = np.tile([18.0, 3.0], (len(times), 1))
    los = truth[0] / np.linalg.norm(truth[0])
    r = np.linalg.norm(truth[0])
    pitch = np.radians(0.3) * np.sin(2 * np.pi * times / 8.0)  # 0.3 deg, 8 s period
    for seed in range(20):
        raw = _observe(truth, egos, np.random.default_rng(400 + seed))
        raw = raw + (r * r * pitch / V.DEFAULT_CONFIG.camera_height)[:, None] * los
        m = V.smooth_vru(times, raw, egos, "person")
        assert m.stationary(), (seed, m.stationary_report())


def test_walking_pedestrian_speed():
    times = _times()
    egos = _ego_path(times, speed=0.0)
    truth = np.stack([np.full_like(times, 15.0), -7.0 + 1.4 * times], axis=1)  # crossing at 1.4 m/s
    speeds = []
    for seed in range(20):
        m = V.smooth_vru(times, _observe(truth, egos, np.random.default_rng(200 + seed)), egos, "person")
        speeds.append(m.speed())
        assert not m.stationary(), (seed, m.stationary_report())
    assert abs(np.median(speeds) - 1.4) < 0.25, speeds
    assert max(abs(s - 1.4) for s in speeds) < 0.5, speeds


def test_walking_along_sight_is_not_called_standing():
    """Walking straight away from the camera is the hard direction (r^2 noise)."""
    times = _times()
    egos = _ego_path(times, speed=0.0)
    truth = np.stack([12.0 + 1.4 * times, np.full_like(times, 2.0)], axis=1)
    moving = 0
    for seed in range(20):
        m = V.smooth_vru(times, _observe(truth, egos, np.random.default_rng(300 + seed)), egos, "person")
        moving += not m.stationary()
    assert moving >= 18, moving


def test_outlier_is_rejected():
    times = _times()
    egos = _ego_path(times, speed=0.0)
    truth = np.stack([np.full_like(times, 15.0), -7.0 + 1.4 * times], axis=1)
    rng = np.random.default_rng(7)
    raw = _observe(truth, egos, rng)
    clean = V.smooth_vru(times, raw, egos, "person")
    bad = raw.copy()
    k = len(times) // 2
    bad[k] += [0.0, 6.0]  # 6 m sideways jump: a mis-association, not range noise
    m = V.smooth_vru(times, bad, egos, "person")
    assert not m.inlier[k]
    assert m.inlier.sum() >= len(times) - 2
    assert np.max(np.linalg.norm(m.positions - clean.positions, axis=1)) < 0.3
    assert abs(m.speed() - 1.4) < 0.4


def test_speed_cap_by_class():
    times = _times()
    egos = _ego_path(times, speed=0.0)
    truth = np.stack([np.full_like(times, 15.0), -20.0 + 5.0 * times], axis=1)  # 5 m/s
    raw = _observe(truth, egos, np.random.default_rng(11))
    ped = V.smooth_vru(times, raw, egos, "person")
    bike = V.smooth_vru(times, raw, egos, "cyclist")
    assert np.max(np.linalg.norm(ped.velocities, axis=1)) <= V.DEFAULT_CONFIG.pedestrian.max_speed + 1e-9
    assert abs(bike.speed() - 5.0) < 0.6, bike.speed()


def test_noise_model_shape():
    along10, lat10 = V.measurement_sigmas(10.0)
    along20, lat20 = V.measurement_sigmas(20.0)
    assert along20 > lat20 * 3  # along-sight dominates at range
    assert 3.5 < along20 / along10 < 4.5  # ~ r^2
    assert V.measurement_sigmas(20.0, correlated=True)[0] > along20  # pitch adds to the stationary test
    assert lat20 > lat10


def test_track_adapter_and_annotation():
    times = _times()
    poses = [{"x": 0.0, "y": 0.0, "yaw": 0.0, "t": float(t)} for t in times]
    rng = np.random.default_rng(3)
    truth = np.tile([20.0, 4.0], (len(times), 1))
    raw = _observe(truth, np.zeros((len(times), 2)), rng)
    ped = {"kind": "person", "samples": [{"token": str(k), "world": list(raw[k]), "box": list(raw[k])}
                                         for k in range(len(times))]}
    car = {"kind": "car", "samples": [{"token": str(k), "world": [1.0, 2.0]} for k in range(len(times))]}
    V.keep_raw_world([ped, car])
    V.annotate_vru_tracks([ped, car], poses, 1.0 / HZ, replace_world=True)
    assert ped["vru_motion"]["stationary"]
    assert ped["vru_motion"]["speed"] < 0.8
    assert "world_raw" in ped["samples"][0] and "world_vru" in ped["samples"][0]
    # Standing -> held at one point, like smooth_world holds a parked car.
    held = {tuple(sample["world"]) for sample in ped["samples"]}
    assert len(held) == 1
    assert np.linalg.norm(np.array(held.pop()) - [20.0, 4.0]) < 1.0
    assert "vru_motion" not in car and "world_vru" not in car["samples"][0]


def test_clip_bounds_default():
    assert V.clip_bounds(30.0) == (10.0, 45.0)
    assert V.clip_bounds(12.0, video_duration=40.0) == (0.0, 27.0)
    cfg = V.ClipWindowConfig(pre_event_s=10.0, post_event_s=10.0)
    assert V.clip_bounds(30.0, config=cfg) == (20.0, 40.0)


def test_tag_window_clamped_instead_of_undecided():
    # Old clips: event +- 10 s, turn ~10 s in -> window starts at frame 0.
    w = V.clamp_tag_window(turn_start=9.5, turn_end=12.0, clip_start=0.0, clip_end=20.0)
    assert w["decided"] and w["start"] == 0.0 and w["clamped"]
    assert w["coverage"] > 0.7
    # Mostly outside the clip -> still undecided.
    w = V.clamp_tag_window(turn_start=1.0, turn_end=2.0, clip_start=0.0, clip_end=6.0)
    assert not w["decided"] and w["reason"] == "segment_boundary"
    # Fully inside.
    w = V.clamp_tag_window(turn_start=20.0, turn_end=23.0, clip_start=0.0, clip_end=35.0)
    assert w["decided"] and not w["clamped"] and w["coverage"] == 1.0


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except AssertionError as exc:
                failed += 1
                print("FAIL", name, exc)
    sys.exit(1 if failed else 0)
