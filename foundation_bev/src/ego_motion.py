"""iter3 visual ego-motion: ground-plane LK flow -> per-frame (speed, yaw rate) -> curved ego path.

Features: Shi-Tomasi corners inside the road mask (or a below-horizon trapezoid if no mask), with
object boxes removed. Each tracked pair is lifted to the BEV through the flat-ground model, and a
rigid 2D transform (yaw + translation) is fitted with RANSAC: points p_prev map to p_cur = R p_prev + t.
The ego moved by the inverse: forward ds = -t_z, yaw dpsi = +angle(R) (iter15: + = RIGHT turn, the convention of
the path integration x += ds*sin(psi), ego_to_world / world_to_ego and lines_mt._to_frame; iter3-14 used -angle(R),
i.e. + = left, so every consumer mirrored the turn: KITTI lane_split, a 11 deg/s right turn in OXTS, drew a left turn).
Output is smoothed (median-of-5, then a centred window) and falls back to the prior speed
when there are too few inliers.
"""
import numpy as np
import cv2


def _road_feature_mask(h, w, horizon_v, road=None, boxes=(), y0f=0.05, y1f=0.97):
    m = np.zeros((h, w), np.uint8)
    if road is not None:
        m[road > 0] = 255
    else:
        top = int(horizon_v + y0f * (h - horizon_v)) + 8
        pts = np.array([[int(0.30 * w), top], [int(0.70 * w), top], [w - 1, int(y1f * h)], [0, int(y1f * h)]], np.int32)
        cv2.fillPoly(m, [pts], 255)
    m[: int(horizon_v) + 8] = 0
    m[int(y1f * h):] = 0
    for b in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b[:4]]
        m[max(0, y0 - 4):y1 + 4, max(0, x0 - 4):x1 + 4] = 0
    return m


def rigid2d_ransac(a, b, thr=0.25, iters=200, rng=None):
    """Fit b ~= R a + t (2D, rotation only). a,b: (N,2) as (x,z). Returns (theta, t, inlier mask)."""
    rng = rng or np.random.default_rng(0)
    n = len(a)
    if n < 6:
        return None
    best = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        da, db = a[j] - a[i], b[j] - b[i]
        if np.linalg.norm(da) < 0.5:
            continue
        th = np.arctan2(db[1], db[0]) - np.arctan2(da[1], da[0])
        th = (th + np.pi) % (2 * np.pi) - np.pi
        if abs(th) > 0.2:
            continue
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        t = b[i] - R @ a[i]
        r = np.linalg.norm(b - (a @ R.T + t), axis=1)
        inl = r < thr
        if best is None or inl.sum() > best[2].sum():
            best = (th, t, inl)
    if best is None or best[2].sum() < 6:
        return None
    # least-squares refit (Kabsch 2D) on inliers
    A, B = a[best[2]], b[best[2]]
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    th = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = cb - R @ ca
    return th, t, best[2]


def med_then_mean(x, k=2):
    """median-of-(2k+1) then centred mean-of-(2k+1), edge padded (the EgoMotion.finalize smoother)."""
    p = np.pad(x, k, mode="edge")
    med = np.array([np.median(p[i:i + 2 * k + 1]) for i in range(len(x))])
    p = np.pad(med, k, mode="edge")
    return np.array([p[i:i + 2 * k + 1].mean() for i in range(len(x))])


class EgoMotion:
    def __init__(self, gm, h, w, fps, prior_speed, max_speed=45.0, zmax=30.0):
        self.gm, self.h, self.w, self.fps = gm, h, w, fps
        self.prior, self.max_speed, self.zmax = prior_speed, max_speed, zmax
        self.prev = None
        self.guess_fracs = (0.0, 0.3, 0.6, 1.0, 1.4)
        self.raw = []  # per frame transition: (ds_m, dpsi_rad, n_inliers) or None

    def step(self, bgr, road=None, boxes=()):
        g_cur = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev = (g_cur, road, boxes)
            return None
        g0, road0, boxes0 = self.prev
        m = _road_feature_mask(self.h, self.w, self.gm.horizon_v, road0, boxes0)
        p0 = cv2.goodFeaturesToTrack(g0, 400, 0.005, 6, mask=m, blockSize=7)
        res = None
        if p0 is not None and len(p0) >= 8:
            A0 = self.gm.lift(p0[:, 0, 0], p0[:, 0, 1])
            fin = np.isfinite(A0).all(1) & (A0[:, 1] < self.zmax)
            p0, A0 = p0[fin], A0[fin]
            # Large per-frame ground motion (25 m/s at 4 fps = 6 m) breaks plain LK, which then locks
            # onto features moving with the ego (cars ahead). Seed LK with flow predicted from a set of
            # candidate speeds and keep the hypothesis with the most RANSAC inliers.
            for g in self.guess_fracs:
                if len(p0) < 8:
                    break
                dsg = g * self.prior / self.fps
                pg = self.gm.project(np.stack([A0[:, 0], A0[:, 1] - dsg], 1)).astype(np.float32)
                okg = np.isfinite(pg).all(1) & (pg[:, 1] < self.h + 50)
                if okg.sum() < 8:
                    continue
                q0 = p0[okg].copy(); qg = pg[okg].reshape(-1, 1, 2).copy()
                p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g_cur, q0, qg, winSize=(21, 21), maxLevel=3,
                                                     flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
                pb, _, _ = cv2.calcOpticalFlowPyrLK(g_cur, g0, p1, q0.copy(), winSize=(21, 21), maxLevel=3,
                                                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
                ok = (st[:, 0] == 1) & (np.linalg.norm((pb - q0)[:, 0], axis=1) < 1.0)
                if ok.sum() < 8:
                    continue
                a, b = q0[ok, 0], p1[ok, 0]
                A = self.gm.lift(a[:, 0], a[:, 1]); B = self.gm.lift(b[:, 0], b[:, 1])
                v = np.isfinite(A).all(1) & np.isfinite(B).all(1)
                if v.sum() < 8:
                    continue
                thr = 0.15 + 0.01 * np.median(A[v, 1])
                r = rigid2d_ransac(A[v], B[v], thr=thr)
                if r is None:
                    continue
                th, t, inl = r
                ds, dpsi = -t[1], th  # iter15 sign fix: a right turn rotates static ground points by +angle in (x, z)
                if 0 <= ds <= self.max_speed / self.fps * 1.2 and (res is None or inl.sum() > res[2]):
                    res = (float(ds), float(dpsi), int(inl.sum()))
        self.raw.append(res)
        self.prev = (g_cur, road, boxes)
        return res

    def finalize(self, win=5):
        """Return per-frame arrays (len = n_frames): speed_mps, yaw_rate, valid flag, travelled s, heading psi, x/z path."""
        n = len(self.raw) + 1
        ds = np.full(n, np.nan); dp = np.full(n, np.nan)
        for i, r in enumerate(self.raw):
            if r is not None and r[2] >= 10:
                ds[i + 1], dp[i + 1] = r[0], r[1]
        valid = np.isfinite(ds)
        frac = valid.mean() if n > 1 else 0.0
        fill_ds = np.nanmedian(ds) if valid.sum() >= 3 else self.prior / self.fps
        ds2 = np.where(valid, ds, fill_ds); dp2 = np.where(valid, dp, 0.0)
        ds2[0] = dp2[0] = 0.0
        k = max(1, win // 2)
        ds_s, dp_s = med_then_mean(ds2, k), med_then_mean(dp2, k)
        ds_s[0] = dp_s[0] = 0.0
        psi = np.cumsum(dp_s)
        x = np.cumsum(ds_s * np.sin(psi)); z = np.cumsum(ds_s * np.cos(psi))
        return {"speed_mps": ds_s * self.fps, "yaw_rate": dp_s * self.fps, "valid": valid, "valid_frac": float(frac),
                "s": np.cumsum(ds_s), "psi": psi, "x": x, "z": z,
                "speed_median_mps": float(np.median(ds_s[1:]) * self.fps) if n > 1 else self.prior}


def ego_path_in_current_frame(ego, now, back=None, ahead_s=2.0, fps=10.0):
    """Polyline (x,z) in the *current* ego frame: travelled path behind (frames <= now) plus a
    constant-curvature prediction ahead (ahead_s seconds) from the current speed/yaw rate."""
    X, Z, P = ego["x"], ego["z"], ego["psi"]
    i0 = 0 if back is None else max(0, now - back)
    c, s = np.cos(P[now]), np.sin(P[now])
    dx, dz = X[i0:now + 1] - X[now], Z[i0:now + 1] - Z[now]
    behind = np.stack([c * dx - s * dz, s * dx + c * dz], 1)
    v, w = ego["speed_mps"][now], ego["yaw_rate"][now]
    t = np.linspace(0, ahead_s, 21)
    if abs(w) < 1e-4:
        ahead = np.stack([np.zeros_like(t), v * t], 1)
    else:
        R = v / w
        ahead = np.stack([R * (1 - np.cos(w * t)), R * np.sin(w * t)], 1)
    return behind, ahead


def world_to_ego(ego, frame, xw, zw):
    """World (frame-0 ego) coords -> ego frame at `frame`."""
    c, s = np.cos(ego["psi"][frame]), np.sin(ego["psi"][frame])
    dx, dz = xw - ego["x"][frame], zw - ego["z"][frame]
    return c * dx - s * dz, s * dx + c * dz


def ego_to_world(ego, frame, x, z):
    c, s = np.cos(ego["psi"][frame]), np.sin(ego["psi"][frame])
    return ego["x"][frame] + c * x + s * z, ego["z"][frame] - s * x + c * z


# ---------------------------------------------------------------------------------------------- iter4
# iter3 bug: the flow speed was 0.1-4 m/s against 10-25 m/s priors. Two causes, found on the frames:
#  1. time base: the KITTI clips are 10 Hz captures stored at 4 fps, so speed = ds * file_fps under-reads
#     by 2.5x. iter4 uses cam["capture_hz"] (KITTI 10) for every physical rate (ds/dt, Kalman dt, accel,
#     static test, trail length); the file fps is only used to write the video.
#  2. duplicated frames: munich.mp4 repeats every frame (30 fps file, 15 Hz content), so every other
#     pair has zero motion and the median flow collapses to ~0. iter4 detects repeats (mean |diff| < 0.5)
#     and estimates motion only across distinct frames.
#  On top, the seeded LK chose the 0-speed hypothesis on periodic highway texture. iter4 measures the
#  ground translation directly: phase correlation of consecutive high-passed IPM (BEV) images of the road
#  band, masked by burned-in HUD/overlay pixels. Translation is in BEV pixels -> metres via the IPM
#  resolution, so no depth unit enters.

def bev_ecc_yaw(prev, cur, sx, sy, iters=60):
    """iter15: ego yaw between two road-masked BEV ROIs: Euclidean ECC (rotation + translation) initialised with the
    phase-correlation shift. BEV rows run against z, so the image angle phi is -angle in (x, z); a right turn rotates
    static ground by +angle in (x, z) -> dpsi (+ = right) = -phi. Returns rad or None (ECC failure / |dpsi| > 0.2).
    KITTI OXTS: lane_split mean 11.7 vs 11.1 deg/s right, city_cyclist 0.3 vs 0.6; fc1 VO yaw corr 0.47 / 0.83."""
    W = np.array([[1, 0, sx], [0, 1, sy]], np.float32)
    try:
        _, W = cv2.findTransformECC(prev, cur, W, cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5), None, 5)
    except cv2.error:
        return None
    d = -float(np.arctan2(W[1, 0], W[0, 0]))
    return d if abs(d) < 0.2 else None


def is_repeat(prev_gray, cur_gray, thr=0.5):
    return prev_gray is not None and float(np.abs(cur_gray.astype(np.float32) - prev_gray.astype(np.float32)).mean()) < thr


class BevPhaseEgo:
    """Per-frame forward travel ds (m) from phase correlation of IPM road images. Needs bev_ipm."""

    def __init__(self, gm, capture_hz, prior_speed, z_band=(7.0, 30.0), x_half=6.0, min_resp=0.15, yaw=False):
        import bev_ipm as IP
        self.IP, self.gm = IP, gm
        self.yaw, self.yaw_raw = yaw, []  # iter15: per-transition BEV-ECC yaw (rad, + = right) or None / "repeat"
        self.Hm = IP.ipm_homography(gm)
        self.hz, self.prior = capture_hz, prior_speed
        self.z_band, self.x_half, self.min_resp = z_band, x_half, min_resp
        self.prev, self.prev_gray, self.raw, self.repeat = None, None, [], []

    def _roi(self, bgr, mask=None, keep=None):
        IP = self.IP
        g = cv2.cvtColor(IP.warp(bgr, self.Hm), cv2.COLOR_BGR2GRAY).astype(np.float32)
        u0, v1 = IP.xz_to_bev_px(-self.x_half, self.z_band[0]); u1, v0 = IP.xz_to_bev_px(self.x_half, self.z_band[1])
        sl = (slice(int(v0), int(v1)), slice(int(u0), int(u1)))
        r = g[sl] - cv2.blur(g[sl], (31, 31))
        if mask is not None:
            r[(IP.warp(mask.astype(np.uint8) * 255, self.Hm, interp=cv2.INTER_NEAREST) > 0)[sl]] = 0
        if keep is not None:
            # iter14: keep only the SAM road minus vehicle boxes; the mask edge is feathered (sigma 6 px) because a hard
            # edge is near-static between frames and pulls the phase peak to zero (fc1_merge 4.9 -> 7.2 m/s, GT 7.23)
            kb = (IP.warp(keep.astype(np.uint8) * 255, self.Hm, interp=cv2.INTER_NEAREST) > 0).astype(np.uint8)
            kb = cv2.erode(kb, np.ones((5, 5), np.uint8)).astype(np.float32)
            self.keep_frac = float(kb[sl].mean())
            r = r * cv2.GaussianBlur(kb, (0, 0), 6.0)[sl]
        return r

    def step(self, bgr, hud_mask=None, keep=None):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rep = is_repeat(self.prev_gray, gray)
        self.repeat.append(rep); self.prev_gray = gray
        if rep:
            self.raw.append("repeat")
            self.yaw_raw.append("repeat")
            return None
        self.keep_frac = 1.0
        r = self._roi(bgr, hud_mask, keep)
        out, yw = None, None
        if self.prev is not None and self.keep_frac > 0.05:
            win = cv2.createHanningWindow(r.shape[::-1], cv2.CV_32F)
            (sx, sy), resp = cv2.phaseCorrelate(self.prev, r, win)
            ds = sy * self.IP.RES
            if resp >= self.min_resp and 0.0 <= ds <= 45.0 / self.hz * 1.3:
                out = (float(ds), float(resp))
            if self.yaw:
                yw = bev_ecc_yaw(self.prev, r, sx, sy)
        self.raw.append(out)
        self.yaw_raw.append(yw)
        self.prev = r
        return out

    def finalize_yaw(self, n_frames, win=5):
        """iter15: per-frame yaw rate (rad/s, + = right) and heading psi from the BEV-ECC rotations. A transition across
        repeated file frames is spread evenly over them; failed pairs count as 0; then the EgoMotion smoother."""
        dp = np.zeros(n_frames); pend = 0; nval = ntr = 0
        for i, y in enumerate(self.yaw_raw[:n_frames]):
            if y == "repeat":
                pend += 1
                continue
            if i > 0:
                ntr += 1
            if isinstance(y, float):
                nval += 1
                lo = i - pend
                dp[lo:i + 1] = y / (pend + 1)
            pend = 0
        dp_s = med_then_mean(dp, max(1, win // 2)); dp_s[0] = 0.0
        return {"yaw_rate": dp_s * self.hz, "psi": np.cumsum(dp_s), "valid_frac": nval / max(ntr, 1)}

    def finalize(self, n_frames):
        """Speed per distinct-frame transition, spread over repeated frames; valid fraction; median."""
        ds = [r[0] for r in self.raw if isinstance(r, tuple)]
        n_trans = sum(1 for r in self.raw if r != "repeat") - 1
        valid = len(ds) / max(n_trans, 1)
        # hz = real-time rate of *file* frames; with repeated frames the distinct-frame interval is longer
        distinct = max(1, len(self.raw) - int(sum(self.repeat)))
        med = float(np.median(ds)) * self.hz * distinct / len(self.raw) if ds else float("nan")
        return {"speed_median_mps": med, "valid_frac": float(valid), "n_repeat": int(sum(self.repeat)),
                "capture_hz": self.hz}
