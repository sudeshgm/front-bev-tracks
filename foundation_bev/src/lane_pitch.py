"""iter15: ground-pitch error from lane parallelism, speed-independent (single-frame lane fits, the iter3
fit_frame_lines path: cluster_lateral + ransac_poly2 on lifted SAM lane pixels, no aggregation). Per frame, adjacent
lane fits covering z 10..28 m with 2.6 < width(10) < 4.6 m give r = width(28)/width(10); theta solves
r = (1 + theta*28/h) / (1 + theta*10/h) on the median r (needs >= MIN_PAIRS pairs; clamped to +-1 deg).
theta > 0: the camera looks further down than the ground model assumes (far range and speed over-read)."""
import numpy as np
import bev_smooth as BS

ZN, ZF, MIN_PAIRS, MAX_DEG = 10.0, 28.0, 5, 1.0

def frame_fits(lane_px, gm, min_pts=30, max_z=40.0):
    if not len(lane_px): return []
    b = gm.lift(lane_px[:, 0], lane_px[:, 1]); b = b[np.isfinite(b).all(1)]
    b = b[(np.abs(b[:, 0]) < 12) & (b[:, 1] < max_z)]
    fits = []
    for cl in BS.cluster_lateral(b, None, gap=0.7, min_pts=min_pts):
        f = BS.ransac_poly2(cl, thr=0.2, min_inl=min_pts, min_span=5.0)
        if f is not None: fits.append(f)
    return fits

def frame_ratios(fits):
    X = lambda c, z: c[0] + c[1] * z + c[2] * z * z
    L = sorted([f for f in fits if f["zmin"] <= ZN and f["zmax"] >= ZF], key=lambda f: X(f["c"], ZN))
    out = []
    for a, b in zip(L, L[1:]):
        w10 = X(b["c"], ZN) - X(a["c"], ZN)
        if 2.6 < w10 < 4.6: out.append((X(b["c"], ZF) - X(a["c"], ZF)) / w10)
    return out

def theta_from_ratios(R, h):
    """-> (theta_rad or None, median r, n)"""
    if len(R) < MIN_PAIRS: return None, (float(np.median(R)) if R else float("nan")), len(R)
    r = float(np.median(R)); th = (r - 1.0) * h / (ZF - r * ZN)
    return float(np.clip(th, -np.radians(MAX_DEG), np.radians(MAX_DEG))), r, len(R)
