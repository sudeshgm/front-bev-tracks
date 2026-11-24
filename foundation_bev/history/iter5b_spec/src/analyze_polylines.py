#!/usr/bin/env python3
"""Lane/edge quality stats from <clip>_polylines.json (iter1 format, also iter2).

Per clip: frames, % frames missing any lane / any edge, polylines per frame, points per polyline,
lateral jitter of nearest-left / nearest-right line at z=10 and z=20 m (median / p90 |dx| frame-to-frame),
zig-zag rate (BEV steps with |dx|>1 m and |dx|>|dz|; proxy for edges cutting across cars/rails),
hook rate (pixel polylines whose 2nd or 2nd-last point jumps >40 px off the chain; smooth_poly zero-pad bug),
above-horizon rate (pixel points with v < cy).
"""
import json, sys, glob, math
from pathlib import Path
import numpy as np


def x_at(poly, z):
    a = np.asarray(poly, float)
    if len(a) < 2:
        return None
    a = a[np.isfinite(a).all(1)]
    best = None
    for i in range(len(a) - 1):
        z0, z1 = a[i, 1], a[i + 1, 1]
        if (z0 - z) * (z1 - z) <= 0 and z0 != z1:
            t = (z - z0) / (z1 - z0)
            x = a[i, 0] + t * (a[i + 1, 0] - a[i, 0])
            best = x if best is None or abs(x) < abs(best) else best
    return best


def side_series(frames, key, z):
    L, R = [], []
    for fr in frames:
        xs = [x_at(p, z) for p in fr.get(key, [])]
        xs = [x for x in xs if x is not None]
        l = [x for x in xs if x < 0]
        r = [x for x in xs if x >= 0]
        L.append(max(l) if l else None)
        R.append(min(r) if r else None)
    return L, R


def jitter(series):
    d = [abs(b - a) for a, b in zip(series, series[1:]) if a is not None and b is not None]
    if not d:
        return None, None, 0
    return float(np.median(d)), float(np.percentile(d, 90)), len(d)


def hook(poly_px, thr=40.0):
    a = np.asarray(poly_px, float)
    if len(a) < 5:
        return False
    for i in (1, len(a) - 2):
        mid = 0.5 * (a[i - 1] + a[i + 1])
        if np.linalg.norm(a[i] - mid) > thr:
            return True
    return False


def zigzag(poly_bev):
    a = np.asarray(poly_bev, float)
    if len(a) < 2:
        return 0, 0
    d = np.diff(a, axis=0)
    bad = (np.abs(d[:, 0]) > 1.0) & (np.abs(d[:, 0]) > np.abs(d[:, 1]))
    return int(bad.sum()), len(d)


def analyze(path):
    d = json.load(open(path))
    fr = d["frames"]
    cy = d["K"]["cy"]
    n = len(fr)
    out = {"clip": d["clip"], "route": d.get("prompt_route"), "frames": n}
    for kind in ("lanes", "edges"):
        px, bev = f"{kind}_px", f"{kind}_bev"
        cnt = [len(f.get(px, [])) for f in fr]
        pts = [len(p) for f in fr for p in f.get(px, [])]
        out[f"{kind}_missing_pct"] = round(100.0 * sum(c == 0 for c in cnt) / max(n, 1), 1)
        out[f"{kind}_per_frame"] = round(float(np.mean(cnt)), 2) if cnt else 0
        out[f"{kind}_pts_median"] = int(np.median(pts)) if pts else 0
        hk = [hook(p) for f in fr for p in f.get(px, [])]
        out[f"{kind}_hook_pct"] = round(100.0 * sum(hk) / max(len(hk), 1), 1)
        above = [pt[1] < cy for f in fr for p in f.get(px, []) for pt in p]
        out[f"{kind}_above_horizon_pct"] = round(100.0 * sum(above) / max(len(above), 1), 1)
        zz = [zigzag(p) for f in fr for p in f.get(bev, [])]
        b, t = sum(z[0] for z in zz), sum(z[1] for z in zz)
        out[f"{kind}_zigzag_pct"] = round(100.0 * b / max(t, 1), 1)
        for z in (10, 20):
            L, R = side_series(fr, bev, z)
            for s, ser in (("L", L), ("R", R)):
                med, p90, k = jitter(ser)
                out[f"{kind}_{s}_z{z}_jit_med"] = None if med is None else round(med, 2)
                out[f"{kind}_{s}_z{z}_jit_p90"] = None if p90 is None else round(p90, 2)
                out[f"{kind}_{s}_z{z}_present_pct"] = round(100.0 * sum(x is not None for x in ser) / max(n, 1), 1)
    # implied road width at z=10 between nearest left/right edges
    L, R = side_series(fr, "edges_bev", 10)
    wd = [r - l for l, r in zip(L, R) if l is not None and r is not None]
    out["edge_width_z10_median_m"] = round(float(np.median(wd)), 2) if wd else None
    return out


def main():
    paths = sys.argv[1:] or sorted(glob.glob("polylines/*_polylines.json"))
    rows = [analyze(p) for p in paths]
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
