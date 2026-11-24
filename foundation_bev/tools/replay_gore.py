"""iter11: faithful in-step replay of the gore stage (StageLines.step after the edge tracker):
edges_raw (untrimmed main edges, GPU dump) -> trim_edge (+bridge_gaps) -> KerbState (car boxes from dets_ego)
-> EdgeHold -> retrim_held -> is_flared -> DivergenceGore -> GoreTracker; split edges taken from the polylines.
Lanes: in-step lanes are not dumped; the lane tracker is upstream of every edge change, so a run's own polylines
lanes (minus fallback / backfill lanes) are used as the proxy.
Compares fired frames with the GPU run's own per-frame gores.
usage: replay_gore.py --run DIR [--lanes-run DIR] [--clips a,b] [--set iter9|iter10|cur] [--json out]"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import lines_mt as LM
import bev_smooth as BS
Z = LM.ZS
i10 = int(np.argmin(np.abs(Z - 10.0)))
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--lanes-run"); ap.add_argument("--dets")
ap.add_argument("--clips", default="munich,kitti_highway_exit_approach")
ap.add_argument("--set", default="cur", help="iter9: no gap bridging, no kerb ramp; iter10: 5 m / 0.3 s; cur: env/defaults")
ap.add_argument("--kerb-oracle", help="run dir whose per-frame kerb_occ gates the car boxes (in-step lanes for the traffic guard are not dumped)")
ap.add_argument("--replica", action="store_true", help="use the replica of the iter10 step tail, not StageLines.post_edges")
ap.add_argument("--dump-frames", default="", help="comma list: store the final main_L / main_R per frame (munich-style overlays)")
ap.add_argument("--out", help="write a run dir (polylines + tracks links): replayed main edges -> centre smoothing -> "
                "edge backfill, lanes taken from --lanes-poly (a replay_lanes.py output), for smoothness_metrics.py")
ap.add_argument("--lanes-poly")
ap.add_argument("--json"); ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()
OFF11 = dict(GORE_KERB_RESET=False, EDGE_FRAG_M=0.0, EDGE_BACKFILL_S=0.0)  # iter11 edge-stage switches (absent in iter10 code)
if a.set == "iter9":
    LM.EDGE_GAP_M = 0.0; LM.KERB_RAMP_S = 0.0
    for k_, v_ in OFF11.items(): setattr(LM, k_, v_)
elif a.set == "iter10":
    LM.EDGE_GAP_M = 5.0; LM.KERB_RAMP_S = 0.3
    for k_, v_ in OFF11.items(): setattr(LM, k_, v_)
def xg(p):
    p = np.asarray(p, float); o = np.full(len(Z), np.nan)
    if p.ndim == 2 and len(p) >= 2:
        s = np.argsort(p[:, 1]); p = p[s]
        m = (Z >= p[0, 1]) & (Z <= p[-1, 1]); o[m] = np.interp(Z[m], p[:, 1], p[:, 0])
    return o
def grid(pts):
    o = np.full(len(Z), np.nan)
    for x, z in pts:
        i = int(round((z - Z[0]) / (Z[1] - Z[0])))
        if 0 <= i < len(Z): o[i] = x
    return o
def to_poly(x):
    x = np.asarray(x, float); f = np.isfinite(x) & (Z <= 40.0 + 1e-6)
    if f.sum() < 2:
        return []
    zs = np.arange(Z[f].min(), Z[f].max() + 1e-6, 1.0)
    return np.stack([np.interp(zs, Z[f], x[f]), zs], 1).round(3).tolist()
def write_run(clip, d, eframes, s, fps):
    os.makedirs(f"{a.out}/polylines", exist_ok=True); os.makedirs(f"{a.out}/tracks", exist_ok=True)
    ef = LM.centre_smooth_lines(eframes, s, fps, kind="edges") if LM.LINE_SMOOTH_S > 0 else eframes
    LP = json.load(open(f"{a.lanes_poly}/polylines/{clip}_polylines.json"))["frames"] if a.lanes_poly else None
    if LP is not None and getattr(LM, "EDGE_BACKFILL_S", 0) > 0 and hasattr(LM, "edge_backfill"):
        lanes_f = [[dict(r, xg=xg(p)) for r, p in zip(f_["lanes"], f_["lanes_bev"])] for f_ in LP]
        ef = LM.edge_backfill(ef, s, fps, lanes=lanes_f)
    for k, fr in enumerate(d["frames"]):
        recs, polys = [], []
        for e in ef[k]:
            p = to_poly(e["xg"])
            if len(p) < 2: continue
            recs.append({"id": int(e.get("id", 0)), "side": e.get("side"), "role": e.get("role"), "held": bool(e.get("held")),
                         "hits": int(e.get("hits", 0)), "kerb_occ": bool(e.get("kerb_occ")), "backfill": bool(e.get("backfill")),
                         "flare": bool(e.get("flare")), "frag": bool(e.get("frag_override")),
                         "zmin": p[0][1], "zmax": p[-1][1], "c": [0.0, 0.0, 0.0], "fallback": False})
            polys.append(p)
        fr["edges"], fr["edges_bev"] = recs, polys
        if LP is not None and k < len(LP):
            fr["lanes"], fr["lanes_bev"] = LP[k]["lanes"], LP[k]["lanes_bev"]
    json.dump(d, open(f"{a.out}/polylines/{clip}_polylines.json", "w"))
    t = f"{a.run}/tracks/{clip}_tracks.csv"
    if os.path.exists(t) and not os.path.exists(f"{a.out}/tracks/{clip}_tracks.csv"):
        os.symlink(os.path.realpath(t), f"{a.out}/tracks/{clip}_tracks.csv")
out = {}
for clip in a.clips.split(","):
    d = json.load(open(f"{a.run}/polylines/{clip}_polylines.json")); F = d["frames"]; fps = float(d.get("fps") or 10.0)
    L = json.load(open(f"{a.lanes_run or a.run}/polylines/{clip}_polylines.json"))["frames"]
    de = json.load(open(f"{a.dets or a.run + '/dets_dump'}/{clip}_dets_ego.json"))["frames"]
    K = d["K"]; gm = BS.GroundModel(K["fx"], K["fy"], K["cx"], K["cy"], d["cam_height_m"], d["ground"]["pitch_rad"])
    ego = {k: np.array([e.get("ego", {}).get(k, 0.0) for e in F]) for k in ("x", "z")}
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ego["x"]), np.diff(ego["z"])))])
    kerb = LM.KerbState(fps=fps); hold = LM.EdgeHold(fps=fps); div = LM.DivergenceGore(); gt = LM.GoreTracker(n_birth=3)
    SL = None
    if hasattr(LM.StageLines, "post_edges") and not a.replica:  # iter11+: replay the pipeline's own post-tracker code
        SL = LM.StageLines(fps=fps); kerb, hold, div, gt = SL.kerb, SL.hold, SL.div, SL.gore
    rep, gpu, kocc, gaps, r10, frag = [], [], [], [], [], []
    eframes = []
    dumpf = {int(v) for v in a.dump_frames.split(",") if v.strip()}; dump = {}
    orc = None
    if a.kerb_oracle:
        OF = json.load(open(f"{a.kerb_oracle}/polylines/{clip}_polylines.json"))["frames"]
        orc = {}
        for role in ("main_L", "main_R"):
            on = np.array([any(e.get("role") == role and e.get("kerb_occ") for e in f_["edges"]) for f_ in OF])
            w = np.zeros(len(on), bool)
            for i in np.flatnonzero(on):
                w[max(0, i - LM.KERB_ENTER - 1): i + 1] = True
            orc[role] = w
    for k, fr in enumerate(F):
        lf = L[k]
        st = fr.get("step")
        if st:  # iter11+ dump: the exact in-step lanes
            lanes = [dict(r, xg=grid(r["x"]), kind="lane") for r in st["lanes"]]
        else:
            lanes = [dict(r, xg=xg(p)) for r, p in zip(lf["lanes"], lf["lanes_bev"]) if not r.get("fallback") and not r.get("backfill")]
        boxes = [o_["box"] for o_ in (de[k].get("objs") or []) if o_.get("cls") == "car"] if k < len(de) else []
        cars = LM.car_footprints(boxes, gm.lift)
        ds = float(s[k] - s[k - 1]) if k else 0.0
        ids = {r.get("role"): r for r in fr["edges"]}
        edges = []
        for role, sd in (("main_L", -1), ("main_R", 1)):
            raw = (fr.get("edges_raw") or {}).get(role)
            if raw is None: continue
            xt = LM.trim_edge(grid(raw), sd, lanes)
            if xt is None: continue
            r0 = ids.get(role, {})
            rec = {"x": xt, "id": r0.get("id", 0), "hits": max(int(r0.get("hits", 10)), 10), "held": False}
            edges.append(LM._legacy(rec, side=sd, kind="edge", extra={"role": role, "split": False}))
        if st:
            edges += [dict(r, xg=grid(r["x"]), kind="edge", split=True) for r in st["split"]]
        else:
            for r, p in zip(fr["edges"], fr["edges_bev"]):
                if (r.get("role") or "").startswith("split"):
                    edges.append(dict(r, xg=xg(p), kind="edge"))
        if SL is not None:
            if orc is not None:
                cars_r = {role: (cars if k < len(orc[role]) and orc[role][k] else []) for role in orc}
                ksave = SL.kerb.step
                def kstep(role, x, side, cars_, ref, ds_=0.0, lanes=None, _k=ksave):
                    return _k(role, x, side, cars_r[role], ref, ds_, lanes=lanes)
                SL.kerb.step = kstep
            edges, fg = SL.post_edges(lanes, edges, cars, ds)
            if orc is not None:
                SL.kerb.step = ksave
        else:
            if LM.KERB_CARS:
                mains = {e.get("role"): e for e in edges if e.get("role") in ("main_L", "main_R")}
                for role, sd in (("main_L", -1), ("main_R", 1)):
                    e = mains.get(role)
                    if e is None: continue
                    o = mains.get("main_R" if role == "main_L" else "main_L")
                    cr = cars if orc is None or (k < len(orc[role]) and orc[role][k]) else []
                    xk, app = kerb.step(role, e["xg"], sd, cr, LM._ref_slope(o["xg"] if o is not None else None, lanes), ds, lanes=lanes)
                    if app:
                        e2 = dict(e, xg=xk, kerb_occ=True); edges[edges.index(e)] = e2
            for role, sd in (("main_L", -1), ("main_R", 1)):
                cur = [e for e in edges if e.get("role") == role]
                h = hold.step(role, cur[0] if cur else None, ds)
                h = LM.retrim_held(h, cur[0] if cur else None, sd, lanes)
                if h is not None: h["flare"] = LM.is_flared(h.get("xg"), sd, lanes)
                if cur: edges[edges.index(cur[0])] = h
                elif h is not None: edges.append(h)
            fg = div.step(lanes, edges)
        g = gt.step(fg)
        eframes.append([e for e in edges if e is not None and e.get("role") in ("main_L", "main_R", "split_in", "split_out")])
        mr = [e for e in edges if e and e.get("role") == "main_R"]
        if k in dumpf:
            dump[k] = {e.get("role"): [None if not np.isfinite(v) else round(float(v), 3) for v in np.asarray(e["xg"], float)]
                       for e in edges if e and e.get("role") in ("main_L", "main_R")}
        r10.append(bool(mr and np.isfinite(np.asarray(mr[0]["xg"], float)[i10])))
        frag.append(bool(mr and mr[0].get("frag_override")))
        kocc.append(bool(mr and mr[0].get("kerb_occ")))
        gaps.append(round(div.gaps[1][-1], 2) if div.gaps[1] else None)
        if g: rep.append((fr["frame"], sorted({x.get("src", "?") for x in g})))
        gg = fr.get("gores") or []
        if gg: gpu.append((fr["frame"], sorted({x.get("src", "?") for x in gg})))
    if a.out:
        write_run(clip, d, eframes, s, fps)
    rf = {f for f, _ in rep}; gf = {f for f, _ in gpu}
    out[clip] = {"replay": rep, "gpu": gpu, "match": sorted(rf & gf), "replay_only": sorted(rf - gf), "gpu_only": sorted(gf - rf),
                 "kerb_R": [i for i, v in enumerate(kocc) if v], "gapR": gaps,
                 "R10": [i for i, v in enumerate(r10) if v], "frag_R": [i for i, v in enumerate(frag) if v], "dump": dump}
    print(f"{clip}: replay {rep}\n   gpu {gpu}\n   replay_only {sorted(rf - gf)} gpu_only {sorted(gf - rf)}")
if a.json:
    json.dump(out, open(a.json, "w"))
