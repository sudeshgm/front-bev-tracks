#!/usr/bin/env python3
"""iter2 smoothing library for the foundation BEV pipeline (pure numpy/cv2, CPU-testable).

Contents
- GroundModel: per-clip flat ground with pitch; lift(u,v)->(x,z) and project(x,z)->(u,v).
- estimate_clip_ground(): per-clip pitch / depth-scale from medians over frames (not per frame).
- ObjTracker: ByteTrack-style two-stage association (IoU + BEV distance), constant-velocity
  Kalman in world (x, s) with per-class noise, coast <= 10 frames, RTS smoothing, re-link of
  fragments within 1.5 m, min length 5, majority-vote class, stationary hold.
- Line fitting: thin-structure skeleton, BEV clustering by lateral offset, RANSAC x(z)=c0+c1 z+c2 z^2.
- LineTracker: ego-motion-compensated EMA (alpha 0.3) of coefficients, 0.5 m jump gate,
  hold up to 15 frames, optional centred (zero-phase) window over the track history.
- Edge extraction: drivable-area lower envelope from the ego corridor, object-box masking,
  rail-pair rejection.
Coordinates: BEV x right (m), z forward (m) in the current ego frame. World s = z + ego_travel.
"""
from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # noqa: BLE001
    _lsa = None

# ----------------------------------------------------------------------------- ground model


class GroundModel:
    def __init__(self, fx, fy, cx, cy, height, pitch=0.0, zmin=1.0, zmax=45.0):
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.h = float(height)
        self.pitch = float(pitch)  # + = camera looks down
        self.zmin, self.zmax = zmin, zmax

    @property
    def horizon_v(self):
        return self.cy - self.fy * math.tan(self.pitch)

    def lift(self, us, vs):
        us = np.asarray(us, np.float64).reshape(-1)
        vs = np.asarray(vs, np.float64).reshape(-1)
        x = (us - self.cx) / self.fx
        y = (vs - self.cy) / self.fy
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        Y = y * c + s
        Z = -y * s + c
        out = np.full((len(us), 2), np.nan)
        ok = Y > 1e-4
        t = np.where(ok, self.h / np.where(ok, Y, 1.0), np.nan)
        out[:, 0] = t * x
        out[:, 1] = t * Z
        bad = ~ok | (out[:, 1] < self.zmin) | (out[:, 1] > self.zmax)
        out[bad] = np.nan
        return out

    def project(self, xz):
        xz = np.asarray(xz, np.float64).reshape(-1, 2)
        X, Z = xz[:, 0], xz[:, 1]
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        Y = self.h
        yc = Y * c - Z * s
        zc = Y * s + Z * c
        zc = np.where(zc > 1e-3, zc, np.nan)
        u = self.fx * X / zc + self.cx
        v = self.fy * yc / zc + self.cy
        return np.stack([u, v], 1)


def vp_row_from_lines(lines_px, h):
    """Vanishing row from >=2 roughly straight image lines (list of (N,2) px arrays)."""
    fits = []
    for p in lines_px:
        a = np.asarray(p, np.float64)
        if len(a) < 6 or np.ptp(a[:, 1]) < 0.08 * h:
            continue
        A = np.stack([a[:, 1], np.ones(len(a))], 1)  # u = m v + b
        m, b = np.linalg.lstsq(A, a[:, 0], rcond=None)[0]
        fits.append((m, b))
    rows = []
    for i in range(len(fits)):
        for j in range(i + 1, len(fits)):
            (m1, b1), (m2, b2) = fits[i], fits[j]
            if abs(m1 - m2) < 0.15:  # near parallel in image: no reliable VP
                continue
            v = (b2 - b1) / (m1 - m2)
            if 0 < v < h:
                rows.append(v)
    return rows


def estimate_clip_ground(cam, h, vp_rows, disp_scales, plane_pitches=None, max_pitch_deg=4.0):
    """Per-clip ground: pitch from the median vanishing row over frames (clamped),
    DAV2 metric scale = median over frames (used only for non-ground object fallback)."""
    fy, cy = float(cam["fy"]), float(cam["cy"])
    pitch_vp = None
    if len(vp_rows) >= 5:
        v = float(np.median(vp_rows))
        pitch_vp = math.atan((cy - v) / fy)
    cands = [p for p in (pitch_vp, (float(np.median(plane_pitches)) if plane_pitches else None)) if p is not None]
    pitch = float(np.median(cands)) if cands else 0.0
    lim = math.radians(max_pitch_deg)
    pitch = float(np.clip(pitch, -lim, lim))
    scale = float(np.median(disp_scales)) if len(disp_scales) else None
    return {
        "pitch_rad": pitch,
        "pitch_deg": math.degrees(pitch),
        "pitch_vp_deg": None if pitch_vp is None else math.degrees(pitch_vp),
        "n_vp": len(vp_rows),
        "disp_scale_median": scale,
        "disp_scale_iqr": (float(np.subtract(*np.percentile(disp_scales, [75, 25]))) if len(disp_scales) > 3 else None),
    }


def dav2_plane_pitch(depth, road_mask, fx, fy, cx, cy, band, iters=60, rng=None):
    """RANSAC plane on DAV2-metric road points -> (pitch_rad, height_m) or None. Per frame;
    the caller takes the median over frames."""
    rng = rng or np.random.default_rng(0)
    v0, v1 = int(band[0]), int(band[1])
    ys, xs = np.nonzero(road_mask[v0:v1] > 0)
    if len(xs) < 200:
        return None
    sel = rng.choice(len(xs), size=min(3000, len(xs)), replace=False)
    us, vs = xs[sel].astype(np.float64), (ys[sel] + v0).astype(np.float64)
    z = depth[vs.astype(int), us.astype(int)].astype(np.float64)
    ok = (z > 1.0) & (z < 40)
    if ok.sum() < 150:
        return None
    us, vs, z = us[ok], vs[ok], z[ok]
    P = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], 1)
    best, best_n = None, 0
    for _ in range(iters):
        idx = rng.choice(len(P), 3, replace=False)
        n = np.cross(P[idx[1]] - P[idx[0]], P[idx[2]] - P[idx[0]])
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        d = -n @ P[idx[0]]
        inl = np.abs(P @ n + d) < 0.08
        if inl.sum() > best_n:
            best_n, best = inl.sum(), inl
    if best is None or best_n < 100:
        return None
    Q = P[best]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c)
    n = vt[-1]
    if n[1] < 0:
        n = -n  # normal points down (+y), towards the ground from the camera
    height = float(n @ c)
    # level-frame ground normal (0,1,0) seen from a camera pitched down by p is (0, cos p, sin p)
    pitch = float(math.atan2(n[2], n[1]))
    return pitch, height


# ----------------------------------------------------------------------------- object tracking

CLASS_NOISE = {  # sigma_a (m/s^2) process; meas multipliers (lat, lon)
    "car": (2.5, 1.0, 1.0),
    "person": (0.8, 0.6, 0.6),
    "bicycle": (1.5, 0.7, 0.8),
}
GATE_M = {"car": 3.0, "person": 1.5, "bicycle": 2.0}


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    den = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / den if den > 0 else 0.0


def _assign(cost, max_cost):
    if cost.size == 0:
        return []
    if _lsa is not None:
        r, c = _lsa(np.where(np.isfinite(cost), cost, 1e6))
        return [(i, j) for i, j in zip(r, c) if cost[i, j] <= max_cost]
    out, used_r, used_c = [], set(), set()
    for k in np.argsort(cost, axis=None):
        i, j = np.unravel_index(k, cost.shape)
        if i in used_r or j in used_c or not cost[i, j] <= max_cost:
            continue
        out.append((int(i), int(j)))
        used_r.add(i)
        used_c.add(j)
    return out


def meas_sigma(z, src="ground"):
    z = max(float(z), 1.0)
    lat = 0.15 + 0.02 * z
    lon = min(6.0, max(0.3, 0.05 * z + 0.004 * z * z))  # range error grows ~z^2 (cf. YOLO overlay)
    if src == "wd3d":
        lon *= 0.8
    return lat, lon


class _KF:
    def __init__(self, x, s, cls, dt):
        self.x = np.array([x, s, 0.0, 0.0])
        self.P = np.diag([1.0, 4.0, 4.0, 16.0])
        self.cls = cls
        self.dt = dt

    def FQ(self, steps=1):
        dt = self.dt * steps
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        sa = CLASS_NOISE.get(self.cls, CLASS_NOISE["car"])[0]
        q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * sa * sa
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = q
        Q[np.ix_([1, 3], [1, 3])] = q
        return F, Q

    def predict(self):
        F, Q = self.FQ()
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return F, Q

    def update(self, zx, zs, z_ego, src):
        lat, lon = meas_sigma(z_ego, src)
        m = CLASS_NOISE.get(self.cls, CLASS_NOISE["car"])
        R = np.diag([(lat * m[1]) ** 2, (lon * m[2]) ** 2])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        y = np.array([zx, zs]) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return float(y @ np.linalg.solve(S, y))


class ObjTracker:
    """Online two-stage association; offline finalize() adds RTS smoothing + re-link."""

    def __init__(self, fps, speed, high=0.5, low=0.25, coast=10, min_len=5, relink_m=1.5, relink_gap=10):
        self.dt = 1.0 / max(float(fps), 1e-3)
        self.speed = float(speed)
        self.high, self.low = high, low
        self.coast, self.min_len = coast, min_len
        self.relink_m, self.relink_gap = relink_m, relink_gap
        self.active, self.done = [], []
        self.next_id = 1

    def travel(self, frame):
        return self.speed * frame * self.dt

    def _new(self, frame, d):
        s = d["bev"][1] + self.travel(frame)
        kf = _KF(d["bev"][0], s, d["cls"], self.dt)
        tr = {"id": self.next_id, "kf": kf, "first": frame, "last": frame, "hits": 1, "miss": 0,
              "votes": defaultdict(float), "box": d["box"], "hist": []}
        tr["votes"][d["cls"]] += d["score"]
        F, Q = kf.FQ()
        tr["hist"].append(dict(frame=frame, meas=(d["bev"][0], s), z_ego=d["bev"][1], score=d["score"], src=d.get("src", "ground"),
                               box=d["box"], cls=d["cls"], xf=kf.x.copy(), Pf=kf.P.copy(), xp=kf.x.copy(), Pp=kf.P.copy(), F=F, Q=Q))
        self.next_id += 1
        self.active.append(tr)

    def _cost(self, tr, d, frame, iou_only=False):
        s_meas = d["bev"][1] + self.travel(frame)
        px, ps = tr["kf"].x[0], tr["kf"].x[1]
        dist = math.hypot(px - d["bev"][0], ps - s_meas)
        cls = max(tr["votes"], key=tr["votes"].get)
        gate = GATE_M.get(cls, 3.0) + 0.08 * max(d["bev"][1], 0) + 0.5 * tr["miss"]
        iou = _iou(tr["box"], d["box"])
        if iou_only:
            if iou < 0.3 and dist > 0.6 * gate:
                return np.inf
        elif iou < 0.3 and dist > gate:
            return np.inf
        if {cls, d["cls"]} == {"car", "person"}:
            return np.inf  # never merge car with person
        c = 0.5 * (1 - iou) + 0.5 * min(dist / gate, 1.0)
        if cls != d["cls"]:
            c += 0.2
        return c

    def step(self, frame, dets, has_dets=True):
        """dets: list of {cls, score, box[x1,y1,x2,y2], bev(x,z_ego), src}. has_dets=False on stride frames."""
        for tr in self.active:
            F, Q = tr["kf"].predict()
            tr["_xp"], tr["_Pp"], tr["_F"], tr["_Q"] = tr["kf"].x.copy(), tr["kf"].P.copy(), F, Q
        matched = set()
        if has_dets:
            hi = [i for i, d in enumerate(dets) if d["score"] >= self.high]
            lo = [i for i, d in enumerate(dets) if self.low <= d["score"] < self.high]
            order = sorted(range(len(self.active)), key=lambda k: -self.active[k]["hits"])
            pairs = []
            for stage, pool, iou_only in ((1, hi, False), (2, lo, True)):
                tids = [k for k in order if k not in {p[0] for p in pairs}]
                if not tids or not pool:
                    continue
                C = np.array([[self._cost(self.active[t], dets[i], frame, iou_only) for i in pool] for t in tids])
                for a, b in _assign(C, 0.95):
                    pairs.append((tids[a], pool[b]))
            for t, i in pairs:
                tr, d = self.active[t], dets[i]
                s = d["bev"][1] + self.travel(frame)
                tr["kf"].update(d["bev"][0], s, d["bev"][1], d.get("src", "ground"))
                tr["last"], tr["hits"], tr["miss"], tr["box"] = frame, tr["hits"] + 1, 0, d["box"]
                tr["votes"][d["cls"]] += d["score"]
                tr["hist"].append(dict(frame=frame, meas=(d["bev"][0], s), z_ego=d["bev"][1], score=d["score"], src=d.get("src", "ground"),
                                       box=d["box"], cls=d["cls"], xf=tr["kf"].x.copy(), Pf=tr["kf"].P.copy(),
                                       xp=tr["_xp"], Pp=tr["_Pp"], F=tr["_F"], Q=tr["_Q"]))
                matched.add(t)
            used = {i for _, i in pairs}
            for i in hi:
                if i not in used:
                    self._new(frame, dets[i])
        keep = []
        for k, tr in enumerate(self.active):
            if k in matched or tr["first"] == frame:
                keep.append(tr)
                continue
            tr["miss"] += 1
            tr["hist"].append(dict(frame=frame, meas=None, z_ego=None, score=0.0, src="coast", box=tr["box"], cls=None,
                                   xf=tr["kf"].x.copy(), Pf=tr["kf"].P.copy(), xp=tr["_xp"], Pp=tr["_Pp"], F=tr["_F"], Q=tr["_Q"]))
            if tr["miss"] > self.coast:
                self.done.append(tr)
            else:
                keep.append(tr)
        self.active = keep

    # ---- offline
    @staticmethod
    def _trim(tr):
        h = tr["hist"]
        while h and h[-1]["meas"] is None:
            h.pop()
        return tr

    @staticmethod
    def _rts(hist):
        n = len(hist)
        xs = [h["xf"].copy() for h in hist]
        Ps = [h["Pf"].copy() for h in hist]
        for k in range(n - 2, -1, -1):
            F, Pp = hist[k + 1]["F"], hist[k + 1]["Pp"]
            C = Ps[k] @ F.T @ np.linalg.inv(Pp)
            xs[k] = xs[k] + C @ (xs[k + 1] - hist[k + 1]["xp"])
            Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp) @ C.T
        return np.array(xs)

    def _refilter(self, hist, cls):
        """Re-run the Kalman over a (possibly merged) measurement list so F/Q/Pp are consistent."""
        first = hist[0]
        kf = _KF(first["meas"][0], first["meas"][1], cls, self.dt)
        out = [dict(first, xf=kf.x.copy(), Pf=kf.P.copy(), xp=kf.x.copy(), Pp=kf.P.copy(), F=kf.FQ()[0], Q=kf.FQ()[1])]
        byf = {h["frame"]: h for h in hist}
        for f in range(first["frame"] + 1, hist[-1]["frame"] + 1):
            F, Q = kf.predict()
            xp, Pp = kf.x.copy(), kf.P.copy()
            h = byf.get(f)
            if h is not None and h["meas"] is not None:
                kf.update(h["meas"][0], h["meas"][1], h["z_ego"], h["src"])
                rec = dict(h)
            else:
                rec = dict(frame=f, meas=None, z_ego=None, score=0.0, src="coast", box=None, cls=None)
            rec.update(xf=kf.x.copy(), Pf=kf.P.copy(), xp=xp, Pp=Pp, F=F, Q=Q)
            out.append(rec)
        return out

    def finalize(self, stationary_mps=0.8):
        tracks = [self._trim(t) for t in self.done + self.active]
        self.active, self.done = [], []
        tracks = [t for t in tracks if t["hist"]]
        for t in tracks:
            t["cls"] = max(t["votes"], key=t["votes"].get)
        # re-link fragments (ID-switch suppression): RTS each fragment, propagate A's smoothed end
        # and B's smoothed start to the gap midpoint; join if <= relink_m metres apart or within a
        # 2-dof Mahalanobis gate (chi2 0.99 = 9.21) of the combined covariance.
        def ends(t):
            meas = [h for h in t["hist"] if h["meas"] is not None]
            if len(meas) < 2:
                h = meas[0]
                return h["xf"], h["Pf"], h["xf"], h["Pf"]
            hist = self._refilter(meas, t["cls"])
            xs = self._rts(hist)
            return xs[0], hist[0]["Pf"], xs[-1], hist[-1]["Pf"]

        for t in tracks:
            t["_ends"] = ends(t)
        tracks.sort(key=lambda t: t["hist"][0]["frame"])
        merged = True
        n_relink = 0
        while merged:
            merged = False
            for a in tracks:
                if a.get("dead"):
                    continue
                ea = a["hist"][-1]
                for b in tracks:
                    if b is a or b.get("dead"):
                        continue
                    sb = b["hist"][0]
                    gap = sb["frame"] - ea["frame"]
                    if not (0 < gap <= self.relink_gap) or b["cls"] != a["cls"]:
                        continue
                    xa, Pa = a["_ends"][2], a["_ends"][3]
                    xb, Pb = b["_ends"][0], b["_ends"][1]
                    half = 0.5 * gap * self.dt
                    pa = xa[:2] + xa[2:] * half
                    pb = xb[:2] - xb[2:] * half
                    d = pa - pb
                    S = Pa[:2, :2] + Pb[:2, :2] + np.eye(2) * 0.05
                    maha = float(d @ np.linalg.solve(S, d))
                    z_rng = max(1.0, sb["z_ego"] or 1.0)
                    if math.hypot(*d) <= self.relink_m or (maha <= 9.21 and math.hypot(*d) <= max(4.0, 0.15 * z_rng)):
                        a["hist"] = a["hist"] + b["hist"]
                        for k, v in b["votes"].items():
                            a["votes"][k] += v
                        a["cls"] = max(a["votes"], key=a["votes"].get)
                        a["_ends"] = ends(a)
                        b["dead"] = True
                        merged = True
                        n_relink += 1
                        break
                if merged:
                    break
        tracks = [t for t in tracks if not t.get("dead")]
        out = []
        for t in tracks:
            t["cls"] = max(t["votes"], key=t["votes"].get)
            meas = [h for h in t["hist"] if h["meas"] is not None]
            if len(meas) < self.min_len:
                continue
            hist = self._refilter(meas, t["cls"])
            xs = self._rts(hist)
            pos = xs[:, :2]
            vel = np.linalg.norm(xs[:, 2:], axis=1)
            stationary = bool(np.median(vel) < stationary_mps)
            if stationary:
                pos = np.repeat(np.median(pos, 0, keepdims=True), len(pos), 0)
            rows = []
            for h, p, xf in zip(hist, pos, xs):
                rows.append(dict(frame=h["frame"], meas=h["meas"], filt=(float(h["xf"][0]), float(h["xf"][1])),
                                 smooth=(float(p[0]), float(p[1])), vel=(float(xf[2]), float(xf[3])),
                                 score=h["score"], box=h["box"], src=h["src"], coast=h["meas"] is None))
            out.append({"id": t["id"], "cls": t["cls"], "rows": rows, "stationary": stationary})
        self.n_relink = n_relink
        return out


# ----------------------------------------------------------------------------- thin structures


def thin(mask):
    """Skeleton of a binary mask. cv2.ximgproc / skimage if present, else Zhang-Suen in numpy."""
    m = (mask > 0).astype(np.uint8)
    try:
        import cv2.ximgproc as xi  # type: ignore

        return (xi.thinning(m * 255) > 0).astype(np.uint8)
    except Exception:  # noqa: BLE001
        pass
    try:
        from skimage.morphology import skeletonize  # type: ignore

        return skeletonize(m > 0).astype(np.uint8)
    except Exception:  # noqa: BLE001
        pass
    img = np.pad(m, 1)
    changed = True
    it = 0
    while changed and it < 40:
        changed = False
        it += 1
        for step in (0, 1):
            P = img
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(n.astype(np.int32) for n in nb)
            seq = nb + [p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int32) for i in range(8))
            c = P[1:-1, 1:-1] == 1
            if step == 0:
                cond = c & (B >= 2) & (B <= 6) & (A == 1) & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = c & (B >= 2) & (B <= 6) & (A == 1) & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            if cond.any():
                img[1:-1, 1:-1][cond] = 0
                changed = True
    return img[1:-1, 1:-1]


def thin_bright_mask(bgr, road, y0, y1, horizon_v):
    """White/yellow thin stripes inside the road: top-hat on V plus colour gate, below horizon."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    k = max(9, (w // 40) | 1)
    toph = cv2.morphologyEx(V, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    white = (toph > 25) & (hsv[:, :, 1] < 90) & (V > 120)
    yellow = (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] > 70) & (V > 90) & (toph > 15)
    m = (white | yellow) & (road > 0)
    top = int(max(y0, horizon_v + 6))
    m[:top] = False
    m[int(y1):] = False
    m = m.astype(np.uint8)
    num, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(num, bool)
    for i in range(1, num):
        x, y, ww, hh, area = st[i]
        if area < 8 or area > 0.02 * h * w:
            continue
        if ww > 0.25 * w and hh < 0.05 * h:  # wide horizontal blob: stop line / crossing / shadow
            continue
        keep[i] = True
    return keep[lab].astype(np.uint8)


def static_overlay_mask(frames_bgr):
    """Burned-in HUD/overlay pixels: saturated or very bright with ~zero temporal variance."""
    if len(frames_bgr) < 5:
        return None
    st = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]).astype(np.float32)
    sd = st.std(0)
    hsv = cv2.cvtColor(frames_bgr[len(frames_bgr) // 2], cv2.COLOR_BGR2HSV)
    m = (sd < 4.0) & ((hsv[:, :, 1] > 120) | (hsv[:, :, 2] > 235))
    return cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8))


def rail_mask(gray, gm: GroundModel, y0, y1, h, w, gauge=(1.30, 1.60)):
    """Long straight Hough pairs whose BEV spacing matches the standard rail gauge."""
    band = np.zeros_like(gray)
    top = int(max(y0, gm.horizon_v + 6))
    band[top:int(y1)] = gray[top:int(y1)]
    edges = cv2.Canny(band, 60, 160)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=int(0.22 * h), maxLineGap=8)
    out = np.zeros((h, w), np.uint8)
    if segs is None:
        return out, 0
    segs = np.asarray(segs).reshape(-1, 4).astype(np.float64)  # cv2 4.x (N,1,4) / 5.x (N,4)
    bev = []
    for x1, y1_, x2, y2 in segs:
        p = gm.lift([x1, x2], [y1_, y2])
        bev.append(p)
    n_pairs = 0
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a, b = bev[i], bev[j]
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            da, db = a[1] - a[0], b[1] - b[0]
            if np.linalg.norm(da) < 3 or np.linalg.norm(db) < 3:
                continue
            ang = abs(math.atan2(da[0], da[1]) - math.atan2(db[0], db[1]))
            if ang > math.radians(3):
                continue
            mz = 0.5 * (a[:, 1].mean() + b[:, 1].mean())
            xa = np.interp(mz, sorted(a[:, 1]), a[np.argsort(a[:, 1]), 0])
            xb = np.interp(mz, sorted(b[:, 1]), b[np.argsort(b[:, 1]), 0])
            if gauge[0] <= abs(xa - xb) <= gauge[1]:
                n_pairs += 1
                for s in (segs[i], segs[j]):
                    cv2.line(out, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), 1, 7)
    return out, n_pairs


def fill_road(road, w):
    """Close small gaps and fill interior holes (lane paint, dashes, manholes) in the drivable mask, so the
    lower-envelope scan does not stop at a dash and lane paint lies inside the road."""
    m = (road > 0).astype(np.uint8)
    k = max(5, (w // 40) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3)))
    h_, w_ = m.shape
    ff = np.zeros((h_ + 2, w_ + 2), np.uint8)
    inv = (1 - m).astype(np.uint8)
    seeds = [(x, 0) for x in range(0, w_, 16)] + [(0, y) for y in range(0, h_, 16)] + [(w_ - 1, y) for y in range(0, h_, 16)]
    outside = np.zeros_like(m)
    for sx, sy in seeds:
        if inv[sy, sx] and not outside[sy, sx]:
            tmp = inv.copy()
            ff[:] = 0
            cv2.floodFill(tmp, ff, (sx, sy), 2)
            outside |= (tmp == 2).astype(np.uint8)
    holes = (inv > 0) & (outside == 0)
    m[holes] = 1
    return m


def box_mask(boxes, h, w, pad=4):
    m = np.zeros((h, w), np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(m, (max(0, x1 - pad), max(0, y1 - pad)), (min(w - 1, x2 + pad), min(h - 1, y2 + pad)), 1, -1)
    return m


def edge_points_lower_envelope(road, occl, rails, y0, y1, horizon_v, u_ego, max_jump=0.25):
    """Inner boundary of the drivable area around the ego corridor, scanned bottom-up.
    Object boxes are filled as road (so a parked car does not bend the edge) but the boundary
    points that touch a box are dropped; rail-pair pixels are dropped; scanning stops when the
    run width jumps (mask leak), i.e. only the lower envelope is kept."""
    h, w = road.shape
    R = ((road > 0) | (occl > 0)).astype(np.uint8)
    left, right = [], []
    centre = float(u_ego)
    prev_w = None
    top = int(max(y0, horizon_v + 6))
    started = False
    for v in range(int(y1) - 1, top, -2):
        row = R[v]
        c = int(np.clip(round(centre), 0, w - 1))
        if not row[c]:
            nz = np.flatnonzero(row)
            if nz.size == 0 or abs(int(nz[np.argmin(np.abs(nz - c))]) - centre) > 0.15 * w:
                if started:
                    break
                continue  # rows below the road (bonnet, band margin): keep scanning upwards
            c = int(nz[np.argmin(np.abs(nz - c))])
        started = True
        l = c
        while l > 0 and row[l - 1]:
            l -= 1
        r = c
        while r < w - 1 and row[r + 1]:
            r += 1
        width = r - l
        n_rows = len(left) + len(right)
        # the bottom rows are often cut by the bonnet / band margin, so only test for a leak once the
        # corridor is established; a leak = the run widens going UP (perspective makes it shrink)
        if prev_w is not None and n_rows >= 12 and width > (1 + max_jump) * prev_w + 0.02 * w:
            break
        prev_w = width if prev_w is None else (max(prev_w, width) if n_rows < 12 else 0.7 * prev_w + 0.3 * width)
        centre = 0.5 * (l + r)
        for u, side in ((l, left), (r, right)):
            if u <= 1 or u >= w - 2:
                continue  # image border is not a road edge
            lo, hi = max(0, u - 4), min(w, u + 5)
            if occl[v, lo:hi].any() or rails[v, lo:hi].any():
                continue
            side.append((float(u), float(v)))
    return [np.asarray(left), np.asarray(right)]


# ----------------------------------------------------------------------------- line fitting


def ransac_poly2(pts, thr=0.2, iters=80, min_inl=12, min_span=6.0, rng=None, max_c1=0.4, max_c2=0.012):
    pts = np.asarray(pts, np.float64)
    pts = pts[np.isfinite(pts).all(1)] if len(pts) else pts
    if len(pts) < min_inl or np.ptp(pts[:, 1]) < min_span:
        return None
    rng = rng or np.random.default_rng(0)
    z, x = pts[:, 1], pts[:, 0]
    best, best_n = None, 0
    deg = 2 if np.ptp(z) > 12 else 1
    for _ in range(iters):
        idx = rng.choice(len(pts), deg + 1, replace=False)
        zi = z[idx]
        if np.min(np.abs(np.subtract.outer(zi, zi)) + np.eye(len(zi)) * 1e9) < 1.0:
            continue  # degenerate minimal sample
        c = np.linalg.solve(np.vander(zi, deg + 1), x[idx])
        inl = np.abs(np.polyval(c, z) - x) < thr
        n = int(inl.sum())
        if n > best_n:
            best_n, best = n, inl
    if best is None or best_n < min_inl or np.ptp(z[best]) < min_span:
        return None
    w_ = 1.0 / np.maximum(z[best], 3.0)  # near points are more reliable
    c = np.polyfit(z[best], x[best], deg, w=w_)
    c = np.concatenate([np.zeros(3 - len(c)), c])[::-1]  # -> c0, c1, c2
    if abs(c[1]) > max_c1 or abs(c[2]) > max_c2:
        return None
    res = np.abs(c[0] + c[1] * z[best] + c[2] * z[best] ** 2 - x[best])
    return {"c": c, "zmin": float(z[best].min()), "zmax": float(z[best].max()), "n": best_n,
            "inlier_frac": best_n / len(pts), "rmse": float(np.sqrt(np.mean(res ** 2)))}


def cluster_lateral(pts, prior_c=None, gap=0.7, min_pts=12):
    """Split BEV points into lines by lateral offset after removing a common heading/curvature."""
    pts = np.asarray(pts, np.float64)
    if len(pts) == 0:
        return []
    pts = pts[np.isfinite(pts).all(1)]
    c1, c2 = (prior_c[1], prior_c[2]) if prior_c is not None else (0.0, 0.0)
    off = pts[:, 0] - c1 * pts[:, 1] - c2 * pts[:, 1] ** 2
    order = np.argsort(off)
    off_s = off[order]
    cuts = np.where(np.diff(off_s) > gap)[0] + 1
    out = []
    for grp in np.split(order, cuts):
        if len(grp) >= min_pts:
            out.append(pts[grp])
    return out


def shift_coeffs(c, d):
    """x_new(z) = x_old(z + d): ego moved forward by d metres (zero yaw)."""
    c0, c1, c2 = c
    return np.array([c0 + c1 * d + c2 * d * d, c1 + 2 * c2 * d, c2])


def x_at(c, z):
    return c[0] + c[1] * z + c[2] * z * z


class LineTracker:
    def __init__(self, kind, alpha=0.3, gate=0.5, hold=15, reinit_after=3, z_ref=(10.0, 20.0), max_tracks=6,
                 motion="follow"):
        """motion='follow': the ego follows the road (constant-curvature arc), so a lane is static in
        the ego frame -> identity prediction (default; there is no yaw/odometry for these clips).
        motion='straight': constant world heading, x_new(z) = x_old(z + ds) (only with real odometry)."""
        self.kind, self.alpha, self.gate, self.hold = kind, alpha, gate, hold
        self.motion = motion
        self.reinit_after, self.z_ref, self.max_tracks = reinit_after, z_ref, max_tracks
        self.tracks, self.next_id, self.hist = [], 1, defaultdict(list)
        self.n_gated = 0

    def _dist(self, c_a, c_b, lo, hi):
        zs = [z for z in self.z_ref if lo - 2 <= z <= hi + 2] or [float(np.clip(10.0, lo, hi))]
        return max(abs(x_at(c_a, z) - x_at(c_b, z)) for z in zs)

    def step(self, frame, fits, ds):
        if self.motion == "straight":
            for t in self.tracks:
                t["c"] = shift_coeffs(t["c"], ds)
                t["zmin"] = max(1.0, t["zmin"] - ds)
                t["zmax"] = t["zmax"] - ds
                for p in t["pending"]:
                    p["c"] = shift_coeffs(p["c"], ds)
        used = set()
        cand = []
        for ti, t in enumerate(self.tracks):
            for fi, f in enumerate(fits):
                lo, hi = max(t["zmin"], f["zmin"]), min(max(t["zmax"], t["zmin"] + 5), f["zmax"])
                cand.append((self._dist(t["c"], f["c"], lo, hi), ti, fi))
        cand.sort()
        tused = set()
        for d, ti, fi in cand:
            if ti in tused or fi in used:
                continue
            t, f = self.tracks[ti], fits[fi]
            if d <= self.gate:
                a = self.alpha if t["hits"] >= 2 else 0.6
                t["c"] = (1 - a) * t["c"] + a * f["c"]
                t["zmin"] = (1 - a) * t["zmin"] + a * f["zmin"]
                t["zmax"] = (1 - a) * t["zmax"] + a * f["zmax"]
                t["hits"] += 1
                t["miss"] = 0
                t["pending"] = []
                tused.add(ti)
                used.add(fi)
            elif d <= 3 * self.gate:
                # jump: reject now (gated), re-init only if it persists consistently
                self.n_gated += 1
                pend = t["pending"]
                if pend and self._dist(pend[-1]["c"], f["c"], f["zmin"], f["zmax"]) > self.gate:
                    pend.clear()
                pend.append({"c": f["c"].copy(), "zmin": f["zmin"], "zmax": f["zmax"]})
                if len(pend) >= self.reinit_after:
                    t["c"] = np.mean([p["c"] for p in pend], 0)
                    t["zmin"], t["zmax"] = f["zmin"], f["zmax"]
                    t["pending"] = []
                    t["miss"] = 0
                    t["reinit"] = t.get("reinit", 0) + 1
                tused.add(ti)
                used.add(fi)
        for fi, f in enumerate(fits):
            if fi in used:
                continue
            if any(self._dist(t["c"], f["c"], f["zmin"], f["zmax"]) < 3 * self.gate for t in self.tracks):
                continue
            self.tracks.append({"id": self.next_id, "c": f["c"].copy(), "zmin": f["zmin"], "zmax": f["zmax"],
                                "hits": 1, "miss": 0, "pending": [], "born": frame})
            self.next_id += 1
        for ti, t in enumerate(self.tracks):
            if ti not in tused and t.get("born") != frame:
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= self.hold and t["zmax"] > t["zmin"] + 3]
        self.tracks.sort(key=lambda t: -t["hits"])
        self.tracks = self.tracks[: self.max_tracks * 2]
        live = []
        for t in self.tracks:
            if t["hits"] >= 3:
                rec = {"id": t["id"], "c": t["c"].copy(), "zmin": float(t["zmin"]), "zmax": float(t["zmax"]),
                       "hits": t["hits"], "held": t["miss"] > 0}
                self.hist[t["id"]].append((frame, rec))
                live.append(rec)
        return live[: self.max_tracks]


def centred_smooth(line_hist, ds, win=7):
    """ds = per-frame shift for motion='straight'; pass ds=0 for motion='follow'."""
    """Zero-phase window over one line track: each neighbour's coefficients are motion-shifted
    into the centre frame's ego frame, then averaged. Returns {frame: rec}."""
    frames = [f for f, _ in line_hist]
    recs = {f: r for f, r in line_hist}
    out = {}
    half = win // 2
    for f in frames:
        cs, lo, hi = [], [], []
        for g in range(f - half, f + half + 1):
            r = recs.get(g)
            if r is None:
                continue
            # neighbour g is in ego frame g; centre frame f is (f-g)*ds further ahead
            cs.append(shift_coeffs(r["c"], (f - g) * ds))
            lo.append(r["zmin"] - (f - g) * ds)
            hi.append(r["zmax"] - (f - g) * ds)
        rec = dict(recs[f])
        rec["c"] = np.mean(cs, 0)
        rec["zmin"] = max(1.0, float(np.median(lo)))
        rec["zmax"] = float(np.median(hi))
        out[f] = rec
    return out


def sample_curve(rec, step=1.0, zcap=40.0):
    lo, hi = max(1.0, rec["zmin"]), min(zcap, rec["zmax"])
    if hi - lo < 2:
        return np.zeros((0, 2))
    zs = np.arange(lo, hi + 1e-6, step)
    return np.stack([x_at(rec["c"], zs), zs], 1)
