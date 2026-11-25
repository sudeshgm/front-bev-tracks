"""iter15 unit tests: ego yaw convention (+ = right turn everywhere), the BEV-ECC yaw sign and magnitude on a synthetic
right turn, repeat-frame spreading of the yaw, and the lane-parallelism pitch estimate (round trip through projection)."""
import os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import bev_smooth as BS
import ego_motion as EM
import bev_ipm as IP
import lane_pitch as LP


def test_consumer_convention_psi_positive_is_right():
    ego = {"psi": np.array([0.0, 0.1]), "x": np.zeros(2), "z": np.zeros(2)}
    xw, zw = EM.ego_to_world(ego, 1, 0.0, 1.0)  # ego forward axis in the world
    assert xw > 0 and zw > 0  # heading rotated to the right


def test_rigid_fit_dpsi_matches_world_to_ego_on_right_turn():
    rng = np.random.default_rng(1)
    a = np.stack([rng.uniform(-6, 6, 80), rng.uniform(6, 30, 80)], 1)  # static ground in ego frame k-1
    alpha, ds = np.radians(1.0), 2.0  # right turn + forward travel
    ego = {"psi": np.array([0.0, alpha]), "x": np.array([0.0, ds * np.sin(alpha)]), "z": np.array([0.0, ds * np.cos(alpha)])}
    b = np.stack(EM.world_to_ego(ego, 1, a[:, 0], a[:, 1]), 1)
    th, t, inl = EM.rigid2d_ransac(a, b)
    src = open(EM.__file__).read()
    assert "ds, dpsi = -t[1], th" in src  # EgoMotion uses dpsi = +th
    assert abs(th - alpha) < 1e-3  # +th is the right-turn yaw in the consumer convention


def _bev_turn_pair(alpha, ds, seed=0):
    rng = np.random.default_rng(seed)
    H, W = IP.bev_shape()
    tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 2.0)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xb, zb = IP.bev_px_to_xz(uu, vv)  # current ego frame
    ego = {"psi": np.array([0.0, alpha]), "x": np.array([0.0, ds * np.sin(alpha)]), "z": np.array([0.0, ds * np.cos(alpha)])}
    xa, za = EM.ego_to_world(ego, 1, xb, zb)  # the same ground point in the previous ego frame (= world)
    ua, va = IP.xz_to_bev_px(xa, za)
    cur = cv2.remap(tex, ua.astype(np.float32), va.astype(np.float32), cv2.INTER_LINEAR)
    u0, v1 = IP.xz_to_bev_px(-6.0, 7.0); u1, v0 = IP.xz_to_bev_px(6.0, 40.0)
    sl = (slice(int(v0), int(v1)), slice(int(u0), int(u1)))
    return tex[sl].copy(), cur[sl].copy()


def test_bev_ecc_yaw_right_turn_sign_and_magnitude():
    for a_deg in (1.0, -0.6):
        prev, cur = _bev_turn_pair(np.radians(a_deg), 2.0)
        (sx, sy), _ = cv2.phaseCorrelate(prev, cur, cv2.createHanningWindow(prev.shape[::-1], cv2.CV_32F))
        d = EM.bev_ecc_yaw(prev, cur, sx, sy)
        assert d is not None and abs(np.degrees(d) - a_deg) < 0.1, (a_deg, d)


def test_finalize_yaw_spreads_repeats():
    gm = BS.GroundModel(500.0, 500.0, 320.0, 240.0, 1.5, 0.0)
    e = EM.BevPhaseEgo(gm, 30.0, 10.0, yaw=True)
    e.yaw_raw = [None] + [0.01, "repeat"] * 20  # 15 Hz content in a 30 fps file (munich)
    y = e.finalize_yaw(len(e.yaw_raw))
    mid = y["yaw_rate"][5:-5]
    assert np.allclose(mid, 0.005 * 30.0)  # no 0 / 2x alternation for the median to kill
    assert y["valid_frac"] == 1.0


def test_lane_pitch_round_trip():
    h, th_true = 1.55, np.radians(0.4)
    g_true = BS.GroundModel(500.0, 500.0, 576.0, 480.0, h, th_true)
    g0 = BS.GroundModel(500.0, 500.0, 576.0, 480.0, h, 0.0)
    z = np.arange(6.0, 40.0, 0.2)
    px = np.concatenate([g_true.project(np.stack([np.full_like(z, x0), z], 1)) for x0 in (-5.25, -1.75, 1.75, 5.25)])
    R = []
    for _ in range(6):
        R += LP.frame_ratios(LP.frame_fits(px, g0, 12, 40.0))
    th, r, n = LP.theta_from_ratios(R, h)
    assert n >= 6 and r > 1.0 and abs(np.degrees(th) - 0.4) < 0.08, (n, r, th)
    assert LP.theta_from_ratios(R[:LP.MIN_PAIRS - 1], h)[0] is None


def test_theta_formula_inverse():
    h = 1.65
    for t in (-0.006, 0.0, 0.004):
        r = (1 + t * LP.ZF / h) / (1 + t * LP.ZN / h)
        assert abs(LP.theta_from_ratios([r] * 6, h)[0] - t) < 1e-9
