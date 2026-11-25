"""iter12: full post-tracker replay on an iter11+ GPU run (CPU only, no weights).
Every frame of an iter11+ polylines dump has `edges_raw` (untrimmed main edges) and `step` (the exact in-step lanes
and split edges). This replays, with the CURRENT lines_mt code:
  trim_edge -> StageLines.post_edges (kerb, fragment veto, hold, retrim, flare, divergence gores) -> GoreTracker
  -> lane fallback (taken from the run) + suppress_edge_duplicates
  -> centre_smooth_lines (lanes, edges) -> pair_width_smooth -> lane_backfill -> edge_backfill
and writes a run dir (polylines + linked tracks) for smoothness_metrics.py. Stripe gores are not replayed (none fire).
usage: replay_full.py --run ITER11_RUN --out DIR [--clips a,b] [--json gores.json]"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("FBEV_REPLAY_SRC") or os.path.join(HERE, "..", "src"))  # FBEV_REPLAY_SRC: replay another iteration's code
import lines_mt as LM
import bev_smooth as BS
Z = LM.ZS
ALL = ("kitti_highway_sample,kitti_highway_exit_approach,kitti_highway_late,kitti_highway_final,"
       "kitti_city_cyclist_candidate,kitti_lane_split_candidate,munich,a2d2a,a2d2b")
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--clips", default=ALL); ap.add_argument("--json")
ap.add_argument("--held", default="auto", help="auto: the dumped tracker coast flag (`edges_raw_held`, iter12+ runs), "
                "else off; off: False (iter11 runs do not dump it); gpu: copy the output flag")
a = ap.parse_args()
HELD_MODE = (lambda r0: False) if a.held in ("off", "auto") else (lambda r0: bool(r0.get("held")) and not r0.get("backfill"))


def grid(pts):
    o = np.full(len(Z), np.nan)
    for x, z in pts or []:
        i = int(round((z - Z[0]) / (Z[1] - Z[0])))
        if 0 <= i < len(Z): o[i] = x
    return o


def pack(r):
    return {"id": int(r.get("id") or 0), "side": r.get("side"), "c": [float(v) for v in (r.get("c") if r.get("c") is not None else [0, 0, 0])],
            "zmin": float(r.get("zmin", 0.0)), "zmax": float(r.get("zmax", 0.0)), "hits": int(r.get("hits", 0)),
            "held": bool(r.get("held")), "parent": r.get("parent"), "fork_z": r.get("fork_z"), "fallback": bool(r.get("fallback")),
            "role": r.get("role"), "flare": bool(r.get("flare")), "kerb_occ": bool(r.get("kerb_occ")),
            "backfill": bool(r.get("backfill")), "kerb_alpha": round(float(r.get("kerb_alpha", 0.0)), 3),
            "frag": bool(r.get("frag_override")), "anchored": bool(r.get("anchored")),
            "hold_src": ("edgehold" if "hold_age" in r else "tracker") if r.get("held") else None,
            "car_push": bool(r.get("car_push"))}


def curve(r, gpu_poly=None):
    if r.get("xg") is None:  # fallback lane: the run's own sampled curve
        return np.asarray(gpu_poly if gpu_poly is not None else [], float).reshape(-1, 2)
    return BS.sample_curve(r, 1.0, 40.0)


os.makedirs(f"{a.out}/polylines", exist_ok=True); os.makedirs(f"{a.out}/tracks", exist_ok=True)
summary = {}
for clip in a.clips.split(","):
    d = json.load(open(f"{a.run}/polylines/{clip}_polylines.json")); F = d["frames"]; fps = float(d.get("fps") or 10.0)
    if not all(fr.get("step") is not None for fr in F):
        sys.exit(f"{clip}: no `step` dump (needs an iter11+ run)")
    de = json.load(open(f"{a.run}/dets_dump/{clip}_dets_ego.json"))["frames"]
    K = d["K"]; gm = BS.GroundModel(K["fx"], K["fy"], K["cx"], K["cy"], d["cam_height_m"], d["ground"]["pitch_rad"])
    ego = {k: np.array([e.get("ego", {}).get(k, 0.0) for e in F]) for k in ("x", "z")}
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(ego["x"]), np.diff(ego["z"])))])
    SL = LM.StageLines(fps=fps)
    lanes_live, edges_live, gores_live, fb_poly, cars_live = [], [], [], [], []
    for k, fr in enumerate(F):
        st = fr["step"]
        lanes = [dict(r, xg=grid(r["x"]), kind="lane") for r in st["lanes"]]
        for r in lanes:
            f = np.isfinite(r["xg"])
            r["zmin"], r["zmax"] = (float(Z[f].min()), float(Z[f].max())) if f.any() else (0.0, 0.0)
        boxes = [o_["box"] for o_ in (de[k].get("objs") or []) if o_.get("cls") == "car"] if k < len(de) else []
        cars = LM.car_footprints(boxes, gm.lift); cars_live.append(cars)
        ds = float(s[k] - s[k - 1]) if k else 0.0
        ids = {r.get("role"): r for r in fr["edges"]}
        edges = []
        for role, sd in (("main_L", -1), ("main_R", 1)):
            raw = (fr.get("edges_raw") or {}).get(role)
            if raw is None: continue
            xt = LM.trim_edge(grid(raw), sd, lanes)
            if xt is None: continue
            r0 = ids.get(role, {})
            rh = fr.get("edges_raw_held")
            held = bool(rh.get(role)) if (a.held == "auto" and isinstance(rh, dict)) else HELD_MODE(r0)
            rec = {"x": xt, "id": r0.get("id", 0), "hits": max(int(r0.get("hits", 10)), 10), "held": held}
            edges.append(LM._legacy(rec, side=sd, kind="edge", extra={"role": role, "split": False}))
        edges += [dict(r, xg=grid(r["x"]), kind="edge", split=True) for r in st["split"]]
        edges, fg = SL.post_edges(lanes, edges, cars, ds)
        edges = [e for e in edges if e is not None]
        g = SL.gore.step(fg)
        gores_live.append([{"poly": np.asarray(x["poly"], float).tolist(), "n_stripes": int(x.get("n_stripes", 0)),
                            "src": x.get("src", "?")} for x in g])
        fbp = {}
        for r, p in zip(fr["lanes"], fr["lanes_bev"]):
            if r.get("fallback"):
                lanes.append(dict(r, xg=None)); fbp[(r["side"], r["id"])] = p
        fb_poly.append(fbp)
        lanes = LM.suppress_edge_duplicates(lanes, [e for e in edges if str(e.get("role", "")).startswith("main")])
        lanes_live.append(lanes); edges_live.append(edges)
    if LM.LINE_SMOOTH_S > 0:
        lanes_live = LM.centre_smooth_lines(lanes_live, s, fps, kind="lanes")
        edges_live = LM.centre_smooth_lines(edges_live, s, fps, kind="edges")
    lanes_live = LM.pair_width_smooth(lanes_live, s, fps)
    lanes_live = LM.lane_backfill(lanes_live, s, fps)
    edges_live = LM.edge_backfill(edges_live, s, fps, lanes=lanes_live)
    if hasattr(LM, "edge_anchor_backfill"):  # iter13 (FBEV_REPLAY_SRC may point at older code)
        edges_live = LM.edge_anchor_backfill(edges_live, s, fps, lanes_live)
        edges_live = LM.edge_car_clear(edges_live, cars_live, lanes_live)
    for k, fr in enumerate(F):
        lb = [curve(r, fb_poly[k].get((r.get("side"), r.get("id")))) for r in lanes_live[k]]
        eb = [curve(r) for r in edges_live[k]]
        fr["lanes"] = [pack(r) for r, p in zip(lanes_live[k], lb) if len(p) >= 2]
        fr["lanes_bev"] = [np.round(p, 3).tolist() for p in lb if len(p) >= 2]
        fr["edges"] = [pack(r) for r, p in zip(edges_live[k], eb) if len(p) >= 2]
        fr["edges_bev"] = [np.round(p, 3).tolist() for p in eb if len(p) >= 2]
        fr["gores_gpu"] = fr.get("gores"); fr["gores"] = gores_live[k]
        fr.pop("lanes_px", None); fr.pop("edges_px", None)
    d["version"] = "replay"
    json.dump(d, open(f"{a.out}/polylines/{clip}_polylines.json", "w"))
    t = f"{a.run}/tracks/{clip}_tracks.csv"; dst = f"{a.out}/tracks/{clip}_tracks.csv"
    if os.path.exists(t) and not os.path.exists(dst): os.symlink(os.path.realpath(t), dst)
    rep = [fr["frame"] for fr in F if fr["gores"]]; gpu = [fr["frame"] for fr in F if fr["gores_gpu"]]
    summary[clip] = {"replay": rep, "gpu": gpu,
                     "replay_src": sorted({x["src"] for fr in F for x in fr["gores"]}),
                     "kerb_frames": sum(any(e.get("kerb_occ") for e in fr["edges"]) for fr in F),
                     "held_frames": sum(any(e.get("held") for e in fr["edges"]) for fr in F)}
    print(f"{clip}: gores replay {rep} gpu {gpu} kerb {summary[clip]['kerb_frames']} held {summary[clip]['held_frames']}", flush=True)
if a.json:
    json.dump(summary, open(a.json, "w"))
