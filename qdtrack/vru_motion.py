"""Motion of pedestrians and cyclists for the VRU yield tags.

The vehicle smoother (scripts/overlay.py smooth_world) is tuned for cars: its
process noise allows 3 m/s^2, its range noise is capped at 6 m, and it has no
idea that a person cannot walk at 9 m/s. Run on a standing pedestrian 20 m
away, the monocular range jitter turns into several metres per second of fake
walking, and a standing person looks like someone the ego vehicle yields to.

This module is a separate smoother that the tags use for pedestrian and
cyclist tracks only. Vehicles keep smooth_world.

* Constant-velocity Kalman filter with a forward pass and an RTS backward pass,
  on the raw world positions (ego pose + detected ground contact).
* Process noise and a velocity prior sized from the class's realistic top
  speed (walking ~2.5 m/s, cycling ~8 m/s); the smoothed speed is capped there.
* Measurement noise from the monocular ground-contact model: along the line of
  sight it grows with range squared (a pixel of foot-point error is
  r^2 / (f h) metres), across it grows linearly with range (bearing error) plus
  a box-centre term. Along-sight noise is therefore much larger than lateral.
* Innovation gating: a position whose Mahalanobis distance to the prediction is
  beyond the chi-square gate is dropped (outlier), not averaged in.
* Stationary test: if the net displacement over the window is within
  ``stationary_k`` times the expected single-position noise at that range
  (projected on the displacement direction), the actor is standing.

Everything is plain numpy. All constants live in VruMotionConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PEDESTRIAN_KINDS = ("person", "pedestrian")
CYCLIST_KINDS = ("cyclist", "bicycle", "motorcyclist", "motorcycle")
VRU_KINDS = PEDESTRIAN_KINDS + CYCLIST_KINDS

# chi-square, 2 dof
CHI2_2DOF = {0.95: 5.991, 0.99: 9.210, 0.997: 11.62, 0.999: 13.82}


@dataclass(frozen=True)
class ClassMotion:
    """Physical limits of one actor class."""

    max_speed: float  # m/s, smoothed speed is capped here
    accel_sigma: float  # m/s^2, white-noise acceleration of the CV model


@dataclass(frozen=True)
class VruMotionConfig:
    pedestrian: ClassMotion = field(default_factory=lambda: ClassMotion(max_speed=2.5, accel_sigma=0.6))
    cyclist: ClassMotion = field(default_factory=lambda: ClassMotion(max_speed=8.0, accel_sigma=1.5))
    # Monocular ground-contact noise model (FC1 wide camera defaults).
    focal_px: float = 590.0  # focal length in pixels
    camera_height: float = 1.5  # metres above the road
    foot_px_sigma: float = 2.0  # foot-point (box bottom) pixel noise
    # Slow, time-correlated error: uncompensated camera pitch (braking, road
    # slope). It does not average out over a turn, so it enters the stationary
    # test (and, by default, the filter, see filter_correlated). (Frame-to-frame residuals on the
    # KITTI/A2D2 clips are ~1.4 px along and ~1.6 px across, i.e. the 2 px above.)
    pitch_sigma_deg: float = 0.3
    # The filter also budgets for the slow error by default: modelled as white
    # it is too small and the filter chases the drift up to the speed cap.
    filter_correlated: bool = True
    bearing_px_sigma: float = 2.0  # box-centre horizontal pixel noise
    range_scale_sigma: float = 0.05  # fraction of range; camera-height / scale error (slow, stationary test only)
    lateral_floor: float = 0.15  # metres; body width / box-centre ambiguity
    along_floor: float = 0.25  # metres; minimum along-sight noise up close
    along_cap: float = 15.0  # metres; beyond this the range carries ~no information
    gate_probability: float = 0.997  # innovation gate (chi-square, 2 dof)
    max_consecutive_rejects: int = 4  # then restart the filter at the measurement
    stationary_k: float = 2.0  # displacement < k * expected noise -> standing
    min_samples: int = 3

    def motion_for(self, kind: str | None) -> ClassMotion:
        return self.cyclist if (kind or "").lower() in CYCLIST_KINDS else self.pedestrian


DEFAULT_CONFIG = VruMotionConfig()


def is_vru(kind: str | None) -> bool:
    return (kind or "").lower() in VRU_KINDS


def measurement_sigmas(rng: np.ndarray | float, config: VruMotionConfig = DEFAULT_CONFIG, correlated: bool = False):
    """(sigma_along_sight, sigma_lateral) in metres at range ``rng``.

    Along: a flat-ground foot point at range r moves r^2 / h metres per radian
    of ray-angle error; the angle error is foot-point pixel noise (px / f),
    and, with correlated=True, the slow pitch error plus a range-proportional
    camera-height / scale term (both slow, so only the stationary test uses them).
    Lateral: bearing error r * px / f, plus a fixed body-width floor.
    """
    r = np.maximum(np.asarray(rng, dtype=np.float64), 0.5)
    angle = config.foot_px_sigma / config.focal_px
    along = r * r * angle / config.camera_height
    if correlated:
        angle = np.hypot(angle, np.radians(config.pitch_sigma_deg))
        along = np.hypot(r * r * angle / config.camera_height, config.range_scale_sigma * r)
    along = np.clip(along, config.along_floor, config.along_cap)
    lateral = np.hypot(r * config.bearing_px_sigma / config.focal_px, config.lateral_floor)
    return along, lateral


def measurement_covariance(position, ego, config: VruMotionConfig = DEFAULT_CONFIG,
                           correlated: bool = False) -> np.ndarray:
    """2x2 world-frame covariance of one position seen from ``ego``."""
    delta = np.asarray(position, dtype=np.float64) - np.asarray(ego, dtype=np.float64)
    rng = float(np.hypot(delta[0], delta[1]))
    along, lateral = measurement_sigmas(rng, config, correlated)
    if rng < 1e-6:
        direction = np.array([1.0, 0.0])
    else:
        direction = delta / rng
    normal = np.array([-direction[1], direction[0]])
    return float(along) ** 2 * np.outer(direction, direction) + float(lateral) ** 2 * np.outer(normal, normal)


@dataclass
class VruMotion:
    times: np.ndarray  # (n,)
    raw: np.ndarray  # (n, 2) raw world positions
    positions: np.ndarray  # (n, 2) smoothed
    velocities: np.ndarray  # (n, 2) smoothed
    inlier: np.ndarray  # (n,) bool, False = rejected by the gate
    ranges: np.ndarray  # (n,) metres from the ego
    kind: str
    egos: np.ndarray | None = None  # (n, 2) ego world positions

    def window_mask(self, start: float | None = None, end: float | None = None) -> np.ndarray:
        mask = np.ones(len(self.times), dtype=bool)
        if start is not None:
            mask &= self.times >= start - 1e-9
        if end is not None:
            mask &= self.times <= end + 1e-9
        return mask

    def speed(self, start=None, end=None) -> float:
        """Median smoothed speed (m/s) in the window."""
        mask = self.window_mask(start, end)
        if not mask.any():
            return float("nan")
        return float(np.median(np.linalg.norm(self.velocities[mask], axis=1)))

    def fitted_line(self, idx=None, config: VruMotionConfig = DEFAULT_CONFIG):
        """Generalised-least-squares straight line through the inlier raw
        positions, each weighted by its own range-dependent covariance.
        Returns (centre at the mean time, velocity, mean time)."""
        idx = np.arange(len(self.times)) if idx is None else np.asarray(idx)
        idx = [k for k in idx if self.inlier[k]] or list(idx)
        t = self.times[idx]
        tm = float(np.mean(t))
        A = np.zeros((4, 4))
        b = np.zeros(4)
        for k, tk in zip(idx, t):
            ego = self.egos[k] if self.egos is not None else np.zeros(2)
            # Covariance at the smoothed position: computed at the raw one, a
            # sample thrown long would get a smaller weight (r^2) and bias the
            # fit towards the camera.
            W = np.linalg.inv(measurement_covariance(self.positions[k], ego, config, correlated=config.filter_correlated))
            J = np.array([[1.0, 0.0, tk - tm, 0.0], [0.0, 1.0, 0.0, tk - tm]])
            A += J.T @ W @ J
            b += J.T @ W @ self.raw[k]
        theta = np.linalg.lstsq(A, b, rcond=None)[0]
        return theta[:2], theta[2:], tm

    def _fitted_displacement(self, idx, config: VruMotionConfig) -> np.ndarray:
        """Net displacement over the window from fitted_line. Far less noisy
        than differencing two smoothed end points, each pinned by few samples."""
        idx = np.asarray(idx)
        if len(idx) < 2:
            return np.zeros(2)
        _centre, velocity, _tm = self.fitted_line(idx, config)
        return velocity * float(self.times[idx[-1]] - self.times[idx[0]])

    def displacement(self, start=None, end=None) -> np.ndarray:
        mask = self.window_mask(start, end)
        idx = np.flatnonzero(mask)
        if len(idx) < 2:
            return np.zeros(2)
        return self.positions[idx[-1]] - self.positions[idx[0]]

    def stationary(self, start=None, end=None, config: VruMotionConfig = DEFAULT_CONFIG) -> bool:
        """Net smoothed displacement within ``stationary_k`` x the expected
        single-position noise (projected on the displacement direction, RMS of
        the two window ends). Deliberately per-position, not shrunk by the
        sample count: real monocular errors are correlated in time (ground
        plane, odometry), so a standard-error test would call standing people
        moving."""
        return self.stationary_report(start, end, config)["stationary"]

    def stationary_report(self, start=None, end=None, config: VruMotionConfig = DEFAULT_CONFIG) -> dict:
        mask = self.window_mask(start, end)
        idx = np.flatnonzero(mask)
        if len(idx) < 2:
            return {"stationary": True, "displacement": 0.0, "noise": float("nan"), "duration": 0.0,
                    "net_speed": 0.0, "reason": "too_few_samples"}
        disp = self._fitted_displacement(idx, config)
        dist = float(np.linalg.norm(disp))
        rng = float(np.median(self.ranges[idx]))
        unit = disp / dist if dist > 1e-9 else None
        # Expected single-position noise (random + slow pitch) along the displacement, at each end of
        # the window (its own range and line of sight), combined RMS. The ends
        # matter, not the middle: the ego may close from 35 m to 6 m in a turn.
        centre, velocity, tm = self.fitted_line(idx, config)
        parts = []
        for k in (idx[0], idx[-1]):
            ego = self.egos[k] if self.egos is not None else np.zeros(2)
            cov = measurement_covariance(centre + velocity * (self.times[k] - tm), ego, config, correlated=True)
            if unit is None:
                parts.append(float(np.sqrt(np.linalg.eigvalsh(cov)[0])))
            else:
                parts.append(float(np.sqrt(unit @ cov @ unit)))
        noise = float(np.sqrt(np.mean(np.square(parts))))
        smoothed_disp = float(np.linalg.norm(self.positions[idx[-1]] - self.positions[idx[0]]))
        duration = float(self.times[idx[-1]] - self.times[idx[0]])
        return {
            "stationary": dist < config.stationary_k * noise,
            "displacement": dist,
            "noise": noise,
            "duration": duration,
            "net_speed": dist / duration if duration > 0 else 0.0,
            "range": rng,
            "smoothed_displacement": smoothed_disp,
            "reason": "",
        }


def smooth_vru(times, positions, egos, kind: str = "person", config: VruMotionConfig = DEFAULT_CONFIG) -> VruMotion:
    """Forward Kalman + RTS smoother for one pedestrian/cyclist track.

    times: (n,) seconds, increasing. positions: (n, 2) raw world positions.
    egos: (n, 2) ego world position at each sample (for the line of sight).
    """
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    z = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    e = np.asarray(egos, dtype=np.float64).reshape(-1, 2)
    n = len(t)
    if not (len(z) == n and len(e) == n):
        raise ValueError("times, positions and egos must have the same length")
    if n and np.any(np.diff(t) <= 0):
        raise ValueError("times must increase")
    motion = config.motion_for(kind)
    ranges = np.hypot(*(z - e).T) if n else np.zeros(0)
    if n < config.min_samples:
        vel = np.zeros((n, 2))
        return VruMotion(t, z, z.copy(), vel, np.ones(n, dtype=bool), ranges, kind, e)

    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    gate = CHI2_2DOF.get(config.gate_probability)
    if gate is None:
        gate = float(-2.0 * np.log(1.0 - config.gate_probability))  # chi2 2dof closed form
    # Prior: position from the first measurement, velocity zero with the
    # class's top speed as a 2-sigma bound.
    # Start from the median of the first few positions, so one bad first
    # detection (and its r^2-small noise if thrown short) cannot anchor the
    # track; the first sample itself is then gated like the others.
    head = min(n, 5)
    start = np.median(z[:head], axis=0)
    R0 = measurement_covariance(start, np.median(e[:head], axis=0), config, correlated=config.filter_correlated)
    x = np.array([start[0], start[1], 0.0, 0.0])
    P = np.zeros((4, 4))
    P[:2, :2] = 4.0 * R0
    P[2:, 2:] = np.eye(2) * (motion.max_speed / 2.0) ** 2

    xs_f = np.zeros((n, 4))
    Ps_f = np.zeros((n, 4, 4))
    xs_p = np.zeros((n, 4))
    Ps_p = np.zeros((n, 4, 4))
    Fs = np.zeros((n, 4, 4))
    inlier = np.ones(n, dtype=bool)
    rejects = 0
    for k in range(n):
        if k == 0:
            F = np.eye(4)
            Q = np.zeros((4, 4))
        else:
            dt = t[k] - t[k - 1]
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
            q = motion.accel_sigma ** 2
            Q = q * np.array([
                [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                [dt ** 3 / 2, 0, dt ** 2, 0],
                [0, dt ** 3 / 2, 0, dt ** 2],
            ])
        x = F @ x
        P = F @ P @ F.T + Q
        Fs[k], xs_p[k], Ps_p[k] = F, x, P
        # Noise at the predicted position (not the measured one, whose r^2
        # noise would down-weight long throws and bias towards the camera).
        R = measurement_covariance(x[:2], e[k], config, correlated=config.filter_correlated)
        S = H @ P @ H.T + R
        nu = z[k] - H @ x
        d2 = float(nu @ np.linalg.solve(S, nu))
        if d2 > gate and rejects < config.max_consecutive_rejects:
            inlier[k] = False
            rejects += 1
        elif d2 > gate:
            # Too many rejections in a row: the filter, not the detections, is
            # wrong (correlated monocular range error, or a real jump). Restart
            # from this measurement with the prior velocity, so one bad stretch
            # cannot lock the track out for the rest of the clip.
            rejects = 0
            x = np.array([z[k, 0], z[k, 1], 0.0, 0.0])
            P = np.zeros((4, 4))
            P[:2, :2] = R
            P[2:, 2:] = np.eye(2) * (motion.max_speed / 2.0) ** 2
            # Break the RTS chain here: the restart is not a prediction.
            Ps_p[k] = P + np.eye(4) * 1e6
        else:
            rejects = 0
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ nu
            IKH = np.eye(4) - K @ H
            P = IKH @ P @ IKH.T + K @ R @ K.T  # Joseph form
        # Physical limit inside the filter, not only on the output: a person
        # cannot walk at 5 m/s, so the state is not allowed to drift there and
        # then gate away every measurement that disagrees.
        speed_k = float(np.hypot(x[2], x[3]))
        if speed_k > motion.max_speed:
            x[2:] *= motion.max_speed / speed_k
        xs_f[k], Ps_f[k] = x, P

    xs = xs_f.copy()
    Ps = Ps_f.copy()
    for k in range(n - 2, -1, -1):
        C = Ps_f[k] @ Fs[k + 1].T @ np.linalg.inv(Ps_p[k + 1])
        xs[k] = xs_f[k] + C @ (xs[k + 1] - xs_p[k + 1])
        Ps[k] = Ps_f[k] + C @ (Ps[k + 1] - Ps_p[k + 1]) @ C.T
        speed_k = float(np.hypot(xs[k, 2], xs[k, 3]))
        if speed_k > motion.max_speed:
            xs[k, 2:] *= motion.max_speed / speed_k

    vel = xs[:, 2:].copy()
    speed = np.linalg.norm(vel, axis=1)
    over = speed > motion.max_speed
    vel[over] *= (motion.max_speed / speed[over])[:, None]
    return VruMotion(t, z, xs[:, :2].copy(), vel, inlier, ranges, kind, e)


def smooth_vru_track(track: dict, poses: list[dict], frame_dt: float, kind: str | None = None,
                     config: VruMotionConfig = DEFAULT_CONFIG) -> VruMotion | None:
    """Run smooth_vru on an overlay/qdtrack track dict.

    Uses ``sample["world_raw"]`` when present (set by keep_raw_world), else
    ``sample["world"]``. Predicted (extrapolated) samples are skipped. Time is
    the pose time of the sample's token, else token * frame_dt.
    """
    rows = [s for s in track.get("samples", []) if not s.get("predicted")]
    if not rows:
        return None
    times, worlds, egos = [], [], []
    for sample in rows:
        token = int(sample["token"])
        pose = poses[token]
        times.append(float(pose.get("t", token * frame_dt)))
        worlds.append(sample.get("world_raw", sample["world"]))
        egos.append([pose["x"], pose["y"]])
    order = np.argsort(times, kind="stable")
    times = np.asarray(times)[order]
    keep = np.concatenate([[True], np.diff(times) > 1e-9])
    idx = order[keep]
    return smooth_vru(np.asarray(times)[keep], np.asarray(worlds, dtype=np.float64)[idx],
                      np.asarray(egos, dtype=np.float64)[idx], kind or track.get("kind") or "person", config)


def keep_raw_world(tracks: list[dict]) -> None:
    """Copy each sample's world position to world_raw before any smoothing."""
    for track in tracks:
        for sample in track.get("samples", []):
            if not sample.get("predicted") and "world_raw" not in sample and "world" in sample:
                sample["world_raw"] = [float(v) for v in sample["world"]]


def annotate_vru_tracks(tracks: list[dict], poses: list[dict], frame_dt: float,
                        config: VruMotionConfig = DEFAULT_CONFIG, replace_world: bool = False) -> None:
    """For every VRU track, store the VRU-smoothed motion.

    Adds sample["world_vru"], sample["vel_vru"], sample["vru_outlier"] and
    track["vru_motion"] = {speed, stationary, displacement, noise, ...} over
    the whole track. With replace_world=True, sample["world"] becomes the VRU
    smoothed position (what a tag reading "world" will then see); a track that
    passes the stationary test is held at its median position, as smooth_world
    holds a parked car (at the noise-weighted mean position). Vehicle tracks are untouched.
    """
    for track in tracks:
        if not is_vru(track.get("kind")):
            continue
        motion = smooth_vru_track(track, poses, frame_dt, config=config)
        if motion is None:
            continue
        rows = [s for s in track["samples"] if not s.get("predicted")]
        by_time = {round(float(tt), 6): k for k, tt in enumerate(motion.times)}
        for sample in rows:
            token = int(sample["token"])
            k = by_time.get(round(float(poses[token].get("t", token * frame_dt)), 6))
            if k is None:
                continue
            sample["world_vru"] = [float(v) for v in motion.positions[k]]
            sample["vel_vru"] = [float(v) for v in motion.velocities[k]]
            sample["vru_outlier"] = not bool(motion.inlier[k])
        report = motion.stationary_report(config=config)
        if replace_world:
            hold = motion.fitted_line(config=config)[0] if report["stationary"] else None
            for sample in rows:
                if "world_vru" in sample:
                    sample["world"] = ([float(hold[0]), float(hold[1])] if hold is not None
                                       else list(sample["world_vru"]))
        report["speed"] = motion.speed()
        report["outliers"] = int((~motion.inlier).sum())
        track["vru_motion"] = report


# ---------------------------------------------------------------------------
# Clip and tag windows


@dataclass(frozen=True)
class ClipWindowConfig:
    pre_event_s: float = 20.0  # clip starts this long before the VLA event
    post_event_s: float = 15.0  # and ends this long after
    tag_half_window_s: float = 10.0  # tag looks at turn +- this
    min_window_coverage: float = 0.7  # below this fraction inside the clip -> undecided


DEFAULT_CLIP_WINDOW = ClipWindowConfig()


def clip_bounds(event_t: float, video_duration: float | None = None,
                config: ClipWindowConfig = DEFAULT_CLIP_WINDOW) -> tuple[float, float]:
    """(start, end) seconds of the clip cut around a VLA event, inside the drive."""
    start = max(0.0, float(event_t) - config.pre_event_s)
    end = float(event_t) + config.post_event_s
    if video_duration is not None:
        end = min(end, float(video_duration))
    return start, end


def clamp_tag_window(turn_start: float, turn_end: float, clip_start: float, clip_end: float,
                     config: ClipWindowConfig = DEFAULT_CLIP_WINDOW) -> dict:
    """The tag window (turn +- half window) clamped to the clip.

    Returns {"start", "end", "coverage", "decided", "reason"}. Undecided
    ("segment_boundary") only when less than min_window_coverage of the
    requested window lies inside the clip. The old rule returned
    segment_boundary as soon as the window touched the clip's first frame.
    """
    want_start = float(turn_start) - config.tag_half_window_s
    want_end = float(turn_end) + config.tag_half_window_s
    start = max(want_start, float(clip_start))
    end = min(want_end, float(clip_end))
    length = want_end - want_start
    inside = max(0.0, end - start)
    coverage = inside / length if length > 0 else 0.0
    decided = coverage >= config.min_window_coverage
    return {"start": start, "end": end, "coverage": coverage, "decided": decided,
            "reason": "" if decided else "segment_boundary",
            "clamped": start > want_start or end < want_end}
