"""iter10 offline replay of the lane post-processing (lines_mt.centre_smooth_lines + iter10 pair_width_smooth +
lane_backfill) on saved GPU runs. Lanes come UNSMOOTHED from LANE_SRC (iter8 GPU polylines: same lane pipeline,
no iter9 smoothing); everything else (edges, ego pose, tracks) from BASE (iter9 GPU run).
usage: replay_lanes.py LANE_SRC BASE OUT [--half 0.3] [--pair 0|1] [--backfill 0|1] [--clips a,b]"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lines_mt as LM
Z = LM.ZS
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("base"); ap.add_argument("out")
ap.add_argument("--half", type=float, default=0.3); ap.add_argument("--pair", type=int, default=1)
ap.add_argument("--backfill", type=int, default=1); ap.add_argument("--clips", default="")
a = ap.parse_args()
def xg(p):
    p = np.asarray(p, float); o = np.full(len(Z), np.nan)
    if p.ndim == 2 and len(p) >= 2:
        s = np.argsort(p[:, 1]); p = p[s]
        m = (Z >= p[0, 1] - 1e-6) & (Z <= p[-1, 1] + 1e-6); o[m] = np.interp(Z[m], p[:, 1], p[:, 0])
    return o
def to_poly(x):
    f = np.isfinite(x) & (Z <= 40.0 + 1e-6)
    zs = np.arange(Z[f].min(), Z[f].max() + 1e-6, 1.0) if f.sum() >= 2 else Z[f]
    return np.stack([np.interp(zs, Z[f], x[f]), zs], 1).round(3).tolist() if f.sum() >= 2 else []
os.makedirs(f"{a.out}/polylines", exist_ok=True); os.makedirs(f"{a.out}/tracks", exist_ok=True)
for pj in sorted(glob.glob(f"{a.base}/polylines/*_polylines.json")):
    clip = os.path.basename(pj).replace("_polylines.json", "")
    if a.clips and clip not in a.clips.split(","): continue
    d = json.load(open(pj)); F = d["frames"]; fps = float(d.get("fps") or 10.0)
    S = json.load(open(f"{a.src}/polylines/{clip}_polylines.json"))["frames"]
    ego = {k: np.array([e.get("ego", {}).get(k, 0.0) for e in F]) for k in ("x", "z")}
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ego["x"]), np.diff(ego["z"])))])
    live = []
    for fr in S[:len(F)]:
        row = []
        for r, p in zip(fr["lanes"], fr["lanes_bev"]):
            x = xg(p)
            row.append(dict(r, xg=None if r.get("fallback") else x, _p=p))
        live.append(row)
    if a.half > 0:
        live = LM.centre_smooth_lines(live, s, fps, half_s=a.half, kind="lanes")
    if a.pair:
        live = LM.pair_width_smooth(live, s, fps)
    if a.backfill:
        live = LM.lane_backfill(live, s, fps)
    for k, fr in enumerate(F):
        rows = live[k] if k < len(live) else []
        fr["lanes"] = [{kk: vv for kk, vv in r.items() if kk not in ("xg", "_p")} for r in rows]
        fr["lanes_bev"] = [r["_p"] if r.get("xg") is None else to_poly(np.asarray(r["xg"], float)) for r in rows]
    json.dump(d, open(f"{a.out}/polylines/{clip}_polylines.json", "w"))
    t = f"{a.base}/tracks/{clip}_tracks.csv"; dst = f"{a.out}/tracks/{clip}_tracks.csv"
    if os.path.exists(t) and not os.path.exists(dst): os.symlink(os.path.abspath(t), dst)
