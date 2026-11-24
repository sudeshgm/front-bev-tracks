#!/usr/bin/env python3
"""Replay iter1's raw per-frame BEV lane/edge polylines through the iter2 line fitter + tracker
(RANSAC poly2, EMA 0.3, 0.5 m gate, hold 15, centred 7-frame window). Same raw data, so the
difference in jitter is the iter2 smoothing alone (iter2 extraction/ground fixes come on top).
Writes <out>/polylines/<clip>_polylines.json in iter2 format."""
import argparse, glob, json, sys, os
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import bev_smooth as BS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--iter1", default="/workspace/foundation_bev/iter1")
ap.add_argument("--out", default="replay_iter1")
ap.add_argument("--window", type=int, default=7)
a = ap.parse_args()
meta = json.load(open(f"{a.iter1}/run_meta.json"))["videos"]
Path(a.out, "polylines").mkdir(parents=True, exist_ok=True)
for pj in sorted(glob.glob(f"{a.iter1}/polylines/*_polylines.json")):
    D = json.load(open(pj))
    clip = D["clip"]
    fps = meta[clip]["fps"]
    lt = BS.LineTracker("lane", max_tracks=6)
    et = {s: BS.LineTracker("edge", max_tracks=1) for s in (-1, 1)}
    lanes_live, edges_live = [], []
    for fr in D["frames"]:
        k = fr["frame"]
        lp = [np.asarray(p, float) for p in fr["lanes_bev"] if len(p) >= 2]
        lp = np.concatenate(lp) if lp else np.zeros((0, 2))
        prior = None
        good = [t for t in lt.tracks if t["hits"] >= 3]
        if good:
            prior = np.mean([t["c"] for t in good], 0)
        fits = [f for f in (BS.ransac_poly2(c, thr=0.2, min_inl=10, min_span=5.0) for c in BS.cluster_lateral(lp, prior, min_pts=10)) if f]
        lanes_live.append(lt.step(k, fits, 0.0))
        live_e = []
        for s in (-1, 1):
            ep = [np.asarray(p, float) for p in fr["edges_bev"] if len(p) >= 2]
            ep = np.concatenate(ep) if ep else np.zeros((0, 2))
            ep = ep[ep[:, 0] * s > 0.5] if len(ep) else ep
            f = BS.ransac_poly2(ep, thr=0.35, min_inl=10, min_span=6.0)
            fl = [f] if f is not None and BS.x_at(f["c"], max(f["zmin"], 6.0)) * s > 0.5 else []
            live_e += [dict(r, side=s) for r in et[s].step(k, fl, 0.0)]
        edges_live.append(live_e)
    if a.window > 1:
        sm_l = {i: BS.centred_smooth(h, 0.0, a.window) for i, h in lt.hist.items()}
        sm_e = {(s, i): BS.centred_smooth(h, 0.0, a.window) for s in (-1, 1) for i, h in et[s].hist.items()}
        lanes_live = [[sm_l[r["id"]].get(k, r) for r in L] for k, L in enumerate(lanes_live)]
        edges_live = [[dict(sm_e[(r["side"], r["id"])].get(k, r), side=r["side"]) for r in E] for k, E in enumerate(edges_live)]
    frames = []
    for k in range(len(D["frames"])):
        pk = lambda R: [{"id": int(r["id"]), "side": r.get("side"), "c": [float(v) for v in r["c"]], "zmin": float(r["zmin"]),  # noqa: E731
                         "zmax": float(r["zmax"]), "hits": int(r["hits"]), "held": bool(r["held"])} for r in R]
        frames.append({"frame": k, "lanes": pk(lanes_live[k]), "edges": pk(edges_live[k]),
                       "lanes_bev": [BS.sample_curve(r).round(3).tolist() for r in lanes_live[k] if len(BS.sample_curve(r)) >= 2],
                       "edges_bev": [BS.sample_curve(r).round(3).tolist() for r in edges_live[k] if len(BS.sample_curve(r)) >= 2]})
    json.dump({"clip": clip, "version": "iter1-replay-iter2-lines", "fps": fps, "ego_speed_mps": meta[clip]["ego_speed_mps"],
               "line_motion": "follow", "K": D["K"], "frames": frames}, open(f"{a.out}/polylines/{clip}_polylines.json", "w"))
    print(clip, "lanes tracks", len(lt.hist), "edge tracks", sum(len(t.hist) for t in et.values()), "lane gated", lt.n_gated)
