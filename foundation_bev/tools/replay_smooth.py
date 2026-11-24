"""iter9 offline replay of the centred line smoothing (lines_mt.centre_smooth_lines) on a saved run.
usage: replay_smooth.py SRC_RUN EGO_RUN OUT_RUN [--half 0.2] [--mode pose|shift] [--kinds lanes,edges] [--no-held]
SRC_RUN/polylines/<clip>_polylines.json (e.g. iter8 GPU outputs), EGO_RUN/polylines/<clip>_polylines.json carries the
iter9 per-frame 'ego' pose {x, z, psi} (same video -> same VO / speed). Writes OUT_RUN/polylines + symlinks tracks so
smoothness_metrics.py --run OUT_RUN works."""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lines_mt as LM
import bev_smooth as BS
Z = LM.ZS
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("ego"); ap.add_argument("out")
ap.add_argument("--half", type=float, default=0.3); ap.add_argument("--mode", default="shift")
ap.add_argument("--kinds", default="lanes,edges"); ap.add_argument("--no-held", action="store_true")
ap.add_argument("--clips", default="")
ap.add_argument("--kerb", action="store_true", help="apply the iter9 kerb-behind-cars rule (KerbState) to main edges first, "
                "car boxes from EGO_RUN/polylines/<clip>_dets_ego.json")
ap.add_argument("--no-smooth", action="store_true")
a = ap.parse_args()
def xg(p):
    p = np.asarray(p, float); o = np.full(len(Z), np.nan)
    if p.ndim == 2 and len(p) >= 2:
        s = np.argsort(p[:, 1]); p = p[s]
        m = (Z >= p[0, 1]) & (Z <= p[-1, 1]); o[m] = np.interp(Z[m], p[:, 1], p[:, 0])
    return o
def to_poly(x, ref):
    ref = np.asarray(ref, float); f = np.isfinite(x)
    if not f.any(): return ref.tolist()
    zs = ref[:, 1]; xs = np.interp(zs, Z[f], x[f]); return np.stack([xs, zs], 1).round(3).tolist()
os.makedirs(f"{a.out}/polylines", exist_ok=True); os.makedirs(f"{a.out}/tracks", exist_ok=True)
for pj in sorted(glob.glob(f"{a.src}/polylines/*_polylines.json")):
    clip = os.path.basename(pj).replace("_polylines.json", "")
    if a.clips and clip not in a.clips.split(","): continue
    d = json.load(open(pj)); fps = float(d.get("fps") or 10.0); F = d["frames"]
    eg = json.load(open(f"{a.ego}/polylines/{clip}_polylines.json"))["frames"]
    ego = {k: np.array([e.get("ego", {}).get(k, 0.0) for e in eg[:len(F)]]) for k in ("x", "z", "psi")}
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ego["x"]), np.diff(ego["z"])))])
    nk = 0
    if a.kerb:
        de = json.load(open(f"{a.ego}/polylines/{clip}_dets_ego.json"))["frames"]
        K = d["K"]; gm = BS.GroundModel(K["fx"], K["fy"], K["cx"], K["cy"], d["cam_height_m"], d["ground"]["pitch_rad"])
        ks = LM.KerbState()
        for k, fr in enumerate(F):
            boxes = [o_["box"] for o_ in (de[k].get("objs") or []) if o_.get("cls") == "car"] if k < len(de) else []
            cars = LM.car_footprints(boxes, gm.lift)
            lanes = [dict(r, xg=xg(p)) for r, p in zip(fr["lanes"], fr["lanes_bev"])]
            mains = {}
            for i, r in enumerate(fr["edges"]):
                role = r.get("role") or {-1: "main_L", 1: "main_R"}.get(r.get("side"))
                if role in ("main_L", "main_R"): mains[role] = (i, xg(fr["edges_bev"][i]))
            dsk = float(s[k] - s[k - 1]) if k else 0.0
            for role, sd in (("main_L", -1), ("main_R", 1)):
                if role not in mains:
                    ks.step(role, None, sd, cars, 0.0, dsk); continue
                i, x = mains[role]; o_ = mains.get("main_R" if role == "main_L" else "main_L")
                xk, app = ks.step(role, x, sd, cars, LM._ref_slope(o_[1] if o_ else None, lanes), dsk, lanes=lanes)
                if app:
                    fr["edges_bev"][i] = to_poly(xk, fr["edges_bev"][i]); fr["edges"][i]["kerb_occ"] = True; nk += 1
    print(clip, "kerb frames", nk)
    for kind in ([] if a.no_smooth else a.kinds.split(",")):
        recs = []
        for fr in F:
            row = []
            for i, (r, p) in enumerate(zip(fr[kind], fr[f"{kind}_bev"])):
                rr = dict(r, xg=xg(p), _i=i)
                if kind == "edges" and not rr.get("role"):
                    rr["role"] = {-1: "main_L", 1: "main_R"}.get(r.get("side"))
                if a.no_held and r.get("held"): rr["fallback"] = True  # excluded as target and neighbour
                row.append(rr)
            recs.append(row)
        sm = LM.centre_smooth_lines(recs, s, fps, half_s=a.half, kind=kind, ego=ego if a.mode == "pose" else None)
        for fr, row in zip(F, sm):
            for rr in row:
                fr[f"{kind}_bev"][rr["_i"]] = to_poly(rr["xg"], fr[f"{kind}_bev"][rr["_i"]])
    json.dump(d, open(f"{a.out}/polylines/{clip}_polylines.json", "w"))
    t = f"{a.src}/tracks/{clip}_tracks.csv"; dst = f"{a.out}/tracks/{clip}_tracks.csv"
    if os.path.exists(t) and not os.path.exists(dst): os.symlink(os.path.abspath(t), dst)
    print(clip, "ok")
