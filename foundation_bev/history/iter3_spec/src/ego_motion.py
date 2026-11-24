"""iter3 visual ego-motion: ground-plane LK flow -> per-frame (speed, yaw rate) -> curved ego path.

Features: Shi-Tomasi corners inside the road mask (or a below-horizon trapezoid if no mask), with
object boxes removed. Each tracked pair is lifted to the BEV through the flat-ground model, and a
rigid 2D transform (yaw + translation) is fitted with RANSAC: points p_prev map to p_cur = R p_prev + t.
The ego moved by the inverse: forward ds = -t_z, yaw dpsi = -angle(R).
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
                ds, dpsi = -t[1], -th
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
        def med_then_mean(x):
            p = np.pad(x, k, mode="edge")
            med = np.array([np.median(p[i:i + 2 * k + 1]) for i in range(len(x))])
            p = np.pad(med, k, mode="edge")
            return np.array([p[i:i + 2 * k + 1].mean() for i in range(len(x))])
        ds_s, dp_s = med_then_mean(ds2), med_then_mean(dp2)
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
