"""Offline: run lines_mt.DivergenceGore over a saved polylines JSON (e.g. a GPU run) and print fired frames."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lines_mt as LM
Z = LM.ZS
def xg(p):
    p = np.asarray(p); o = np.full(len(Z), np.nan)
    if len(p) >= 2:
        m = (Z >= p[:, 1].min()) & (Z <= p[:, 1].max()); o[m] = np.interp(Z[m], p[:, 1], p[:, 0])
    return o
for path in sys.argv[1:]:
    d = json.load(open(path)); dg = LM.DivergenceGore(); fired = []; gaps = []
    for f in d["frames"]:
        lanes = [dict(r, xg=xg(p)) for r, p in zip(f["lanes"], f["lanes_bev"])]
        edges = [dict(r, xg=xg(p), role=r.get("role") or {-1: "main_L", 1: "main_R"}.get(r.get("side"))) for r, p in zip(f["edges"], f["edges_bev"])]
        g = dg.step(lanes, edges)
        if g: fired.append((f["frame"], sorted({x["src"] for x in g})))
        gaps.append(round(dg.gaps[1][-1], 2) if dg.gaps[1] else None)
    print(os.path.basename(path), "fired", fired)
    print("  right gap per frame", gaps)
