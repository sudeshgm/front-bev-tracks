"""iter3 detection plausibility gates and track-level draw filters.

Detection level (stage 2, per WD3D box, before association):
  score_ok          per-class score thresholds
  ground_contact    box bottom must sit on the road plane: bottom row below the horizon, ground range at
                    the bottom row consistent with the detector/DAV2 depth, metric height/width within
                    class priors
  on_vertical_plane rejects boxes painted on a vertical planar surface (billboards, posters, bus ads):
                    the depth inside the box lies on the same plane as a ring around it, and that plane
                    is near-vertical
  bicycle_ok        bicycle needs an overlapping/nearby person, or a bicycle-sized box at ground contact
Track level (after RTS smoothing, before drawing):
  track_filter      min hits, track-median score, bicycle rider support, off-road person/bicycle on highways,
                    static flag from world-frame speed (ego motion compensated) -> draw as dot/short stub
"""
import numpy as np

SCORE_THR = {"car": 0.35, "person": 0.45, "bicycle": 0.50}
# metric (height, width) priors in metres: (hmin, hmax, wmin, wmax)
SIZE_PRIOR = {"car": (1.0, 3.6, 1.2, 6.0), "person": (1.0, 2.2, 0.25, 1.4), "bicycle": (0.7, 2.2, 0.3, 2.2)}


def score_ok(cls, score, thr=SCORE_THR):
    return score >= thr.get(cls, 0.4)


def ground_contact(box, cls, gm, depth_z=None, horizon_margin_px=3, rel_tol=0.30, abs_tol=1.5, fy=None, fx=None):
    """box = (x0,y0,x1,y1) px. depth_z = detector (WD3D 3D centre z) or DAV2 metric depth at the box bottom.
    Returns (ok, reason, z_ground)."""
    x0, y0, x1, y1 = box[:4]
    hv = gm.horizon_v
    if y1 <= hv + horizon_margin_px:
        return False, "bottom_above_horizon", None
    p = gm.lift(np.array([0.5 * (x0 + x1)]), np.array([y1]))[0]
    zg = p[1]
    if not np.isfinite(zg):
        zg = gm.fy * gm.h / max(y1 - hv, 1e-3)  # beyond zmax: plain flat-ground range
    if depth_z is not None and np.isfinite(depth_z):
        if abs(depth_z - zg) > max(abs_tol, rel_tol * zg):
            return False, "depth_vs_ground", zg
    fy = fy or gm.fy; fx = fx or gm.fx
    hm = (y1 - y0) * zg / fy
    wm = (x1 - x0) * zg / fx
    hmin, hmax, wmin, wmax = SIZE_PRIOR.get(cls, (0.3, 5.0, 0.2, 8.0))
    # truncated by the image bottom/sides -> only the upper bound on height is meaningful
    if hm > hmax * 1.25:
        return False, "too_tall", zg
    if hm < hmin * 0.6 and y0 > 2:
        return False, "too_short", zg
    if wm > wmax * 1.3:
        return False, "too_wide", zg
    return True, "ok", zg


def _backproject(depth, us, vs, fx, fy, cx, cy):
    z = depth[vs, us]
    return np.stack([(us - cx) * z / fx, (vs - cy) * z / fy, z], 1)


def on_vertical_plane(box, depth, fx, fy, cx, cy, ring=0.8, max_rel_res=0.03, max_ny=0.35, min_ring_inl=0.6, rng=None):
    """True if the box content lies on one near-vertical plane that also continues around it.
    depth: metric (or affine-consistent) depth map, same size as the image."""
    if depth is None:
        return False
    h, w = depth.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    X0, Y0 = max(0, int(x0 - ring * bw)), max(0, int(y0 - ring * bh))
    X1, Y1 = min(w - 1, int(x1 + ring * bw)), min(h - 1, int(y1 + 0.2 * bh))  # little ring below: ground
    if X1 - X0 < 8 or Y1 - Y0 < 8:
        return False
    rng = rng or np.random.default_rng(0)
    uu, vv = np.meshgrid(np.arange(X0, X1, max(1, (X1 - X0) // 40)), np.arange(Y0, Y1, max(1, (Y1 - Y0) // 40)))
    uu, vv = uu.ravel(), vv.ravel()
    inside = (uu >= x0) & (uu < x1) & (vv >= y0) & (vv < y1)
    P = _backproject(depth, uu, vv, fx, fy, cx, cy)
    ok = np.isfinite(P).all(1) & (P[:, 2] > 0.5)
    P, inside = P[ok], inside[ok]
    if inside.sum() < 12 or (~inside).sum() < 12:
        return False
    best = None
    for _ in range(80):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        if abs(n[1]) > max_ny:  # not vertical
            continue
        r = np.abs((P - s[0]) @ n) / np.maximum(P[:, 2], 1.0)
        inl = r < max_rel_res
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None:
        return False
    return bool(best[inside].mean() > 0.85 and best[~inside].mean() > min_ring_inl)


def bicycle_ok(box, persons, gm=None, iou_min=0.05, bev_xz=None, persons_bev=(), bev_dist=1.2, size_ok=False):
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
        return ix * iy / u if u > 0 else 0.0
    if any(iou(box, p) >= iou_min for p in persons):
        return True
    if bev_xz is not None and any(np.hypot(bev_xz[0] - q[0], bev_xz[1] - q[1]) <= bev_dist for q in persons_bev):
        return True
    return bool(size_ok)


def gate_detections(dets, gm, depth=None, K=None, use_plane=True):
    """dets: list of dict(box, cls, score, z (optional detector depth), bev (optional (x,z))).
    Returns (kept, rejected list of (det, reason))."""
    kept, rej = [], []
    for d in dets:
        if not score_ok(d["cls"], d["score"]):
            rej.append((d, "score")); continue
        ok, why, zg = ground_contact(d["box"], d["cls"], gm, d.get("z"))
        if not ok:
            rej.append((d, why)); continue
        d["z_ground"] = zg
        if use_plane and depth is not None and K is not None and d["cls"] in ("person", "bicycle"):
            if on_vertical_plane(d["box"], depth, *K):
                rej.append((d, "vertical_plane")); continue
        kept.append(d)
    persons = [d["box"] for d in kept if d["cls"] == "person"]
    pbev = [d["bev"] for d in kept if d["cls"] == "person" and d.get("bev") is not None]
    out = []
    for d in kept:
        if d["cls"] == "bicycle":
            x0, y0, x1, y1 = d["box"][:4]
            zg = d["z_ground"]
            hm = (y1 - y0) * zg / gm.fy; wm = (x1 - x0) * zg / gm.fx
            size_ok = 0.8 <= hm <= 1.3 and 1.2 <= wm <= 2.0  # side-on riderless bicycle
            if not bicycle_ok(d["box"], persons, bev_xz=d.get("bev"), persons_bev=pbev, size_ok=size_ok):
                rej.append((d, "bicycle_no_rider")); continue
        out.append(d)
    return out, rej


# ----------------------------------------------------------------------------------------------- tracks

def world_speed(frames, x, z, ego, fps):
    """Median world-frame speed of a track. ego: dict with per-frame 's' (travelled) and optionally psi/x/z
    (visual ego-motion). Without psi uses straight motion: z_world = z + s."""
    frames = np.asarray(frames)
    if ego is not None and "psi" in ego:
        from ego_motion import ego_to_world
        W = np.array([ego_to_world(ego, f, xi, zi) for f, xi, zi in zip(frames, x, z)])
        xw, zw = W[:, 0], W[:, 1]
    else:
        s = np.asarray(ego["s"])[frames]
        xw, zw = np.asarray(x), np.asarray(z) + s
    if len(frames) < 2:
        return 0.0, 0.0
    T = (frames[-1] - frames[0]) / fps
    net = float(np.hypot(xw[-1] - xw[0], zw[-1] - zw[0]) / T) if T > 0 else 0.0
    df = np.diff(frames) / fps
    v = np.hypot(np.diff(xw), np.diff(zw)) / np.maximum(df, 1e-6)
    return float(np.median(v)), net


def track_filter(tracks, fps, ego, edges_by_frame=None, highway=False, min_hits=None, static_mps=0.5,
                 thr=SCORE_THR, persons_near_m=1.5, rider_frac=0.3, static_rel=0.08):
    """tracks: {tid: dict(cls, frames, x, z, scores (hits only), hits)}. Returns {tid: decision} with
    decision in {'draw', 'draw_static', 'drop_*'} and a static flag."""
    min_hits = min_hits or max(5 if fps >= 10 else max(2, int(round(0.5 * fps))), int(round(0.5 * fps)))  # iter13: 0.5 s below 10 fps
    out = {}
    # person positions per frame for bicycle rider support
    pmap = {}
    for tid, t in tracks.items():
        if t["cls"] == "person":
            for f, xx, zz in zip(t["frames"], t["x"], t["z"]):
                pmap.setdefault(f, []).append((xx, zz))
    for tid, t in tracks.items():
        if t["hits"] < min_hits:
            out[tid] = "drop_short"; continue
        if np.median(t["scores"]) < thr.get(t["cls"], 0.4):
            out[tid] = "drop_score"; continue
        if t["cls"] == "bicycle":
            sup = np.mean([any(np.hypot(xx - p[0], zz - p[1]) <= persons_near_m for p in pmap.get(f, []))
                           for f, xx, zz in zip(t["frames"], t["x"], t["z"])])
            if sup < rider_frac:
                out[tid] = "drop_bicycle_no_rider"; continue
        if highway and t["cls"] in ("person", "bicycle") and edges_by_frame is not None:
            off = []
            for f, xx, zz in zip(t["frames"], t["x"], t["z"]):
                e = edges_by_frame.get(f)
                if e is None:
                    continue
                lo, hi = e(zz)
                if lo is not None and hi is not None:
                    off.append(xx < lo - 1.0 or xx > hi + 1.0)
            if off and np.mean(off) > 0.5:
                out[tid] = "drop_offroad_vru"; continue
        v_med, v_net = world_speed(t["frames"], t["x"], t["z"], ego, fps)
        # static: world speed < 0.5 m/s, plus a range-scale tolerance proportional to ego speed
        # (an 8% range-scale error leaves a static object with 0.08*v_ego of apparent motion)
        sv = np.diff(np.asarray(ego["s"])[t["frames"][0]:t["frames"][-1] + 1])
        ego_v = float(np.median(sv) * fps) if len(sv) else 0.0
        thr_v = static_mps + static_rel * ego_v
        out[tid] = "draw_static" if v_net < thr_v else "draw"
    return out


def scene_ego_speed(tracks, fps, prior, min_pts=8, min_abs_x=2.0, lo=0.5, hi=1.6):
    """Ego speed from the static majority: robust mode of -dz/dt over off-lane tracks (|x| > min_abs_x:
    mostly parked cars, poles, pedestrians). Cancels the DAV2/flat-ground range-scale error in the static
    test, because the same lift produces both. Returns (speed, n_used, accepted)."""
    v = []
    for t in tracks.values():
        f = np.asarray(t["frames"]); z = np.asarray(t["z"]); x = np.asarray(t["x"])
        if len(f) < min_pts or np.median(np.abs(x)) < min_abs_x:
            continue
        T = (f[-1] - f[0]) / fps
        if T > 0:
            v.append(-(z[-1] - z[0]) / T)
    v = np.asarray(v)
    v = v[(v > lo * prior) & (v < hi * prior)]
    if len(v) < 5:
        return prior, len(v), False
    # mode via a 1 m/s histogram, refined by the median of its neighbourhood
    h, e = np.histogram(v, bins=np.arange(lo * prior, hi * prior + 1.0, 1.0))
    c = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    near = v[np.abs(v - c) < 1.5]
    return float(np.median(near)), int(len(near)), True


def scene_ego_profile(tracks, n_frames, fps, prior, win_s=1.0, min_abs_x=2.0, zmax=40.0, lo=0.3, hi=1.8, min_n=3):
    """Per-frame ego speed from the static majority (sliding window). For each frame, every off-lane
    track alive in [f - W, f + W] with z < zmax contributes its local closing rate -dz/dt; the frame
    speed is the median of those within [lo, hi] x prior (moving traffic is mostly outside that band or a
    minority). Gaps are filled with the clip median (or prior). Returns dict with 's' and 'speed'."""
    W = max(2, int(round(win_s * fps)))
    per = [[] for _ in range(n_frames)]
    for t in tracks.values():
        f = np.asarray(t["frames"]); z = np.asarray(t["z"]); x = np.asarray(t["x"])
        ok = (np.abs(x) >= min_abs_x) & (z < zmax)
        if ok.sum() < 4:
            continue
        f, z = f[ok], z[ok]
        for i in range(len(f)):
            j0 = np.searchsorted(f, f[i] - W); j1 = np.searchsorted(f, f[i] + W, side="right") - 1
            if f[j1] - f[j0] >= W:
                v = -(z[j1] - z[j0]) / ((f[j1] - f[j0]) / fps)
                if lo * prior < v < hi * prior:
                    per[f[i]].append(v)
    sp = np.array([np.median(p) if len(p) >= min_n else np.nan for p in per])
    fill = np.nanmedian(sp) if np.isfinite(sp).sum() >= 5 else prior
    sp = np.where(np.isfinite(sp), sp, fill)
    k = W
    p = np.pad(sp, k, mode="edge")
    sp = np.array([p[i:i + 2 * k + 1].mean() for i in range(n_frames)])
    s = np.concatenate([[0.0], np.cumsum(sp[1:] / fps)])
    return {"s": s, "speed": sp, "valid_frac": float(np.mean([len(q) >= min_n for q in per]))}
