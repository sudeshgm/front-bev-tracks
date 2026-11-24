"""iter7 offline replay of the road-edge gate on GPU polylines.
Input = iter5 GPU polylines (edges BEFORE any heading/crossing gate; iter6's own polylines only keep survivors).
Rules: raw (iter5 as is) | iter6 (drop edge if heading > 15 deg or crosses a lane) | iter7 (trim + 5-frame hold;
lane lines duplicating an edge suppressed). Writes <out>/<rule>/{polylines,tracks} for smoothness_metrics and
prints edge presence at z=10 m plus a residual "diagonal crossing" count."""
import copy, json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lines_mt as LM

Z = LM.ZS
SRC = sys.argv[1]          # dir with polylines/ tracks/ (iter5 GPU)
META = sys.argv[2]         # run_meta.json with selected ego speed (iter6 GPU)
OUT = sys.argv[3]
CLIPS = sys.argv[4].split(",")


def xg(p):
    p = np.asarray(p, float); o = np.full(len(Z), np.nan)
    if len(p) >= 2:
        m = (Z >= p[:, 1].min()) & (Z <= p[:, 1].max()); o[m] = np.interp(Z[m], p[:, 1], p[:, 0])
    return o


def pts(x):
    f = np.isfinite(x)
    return np.stack([x[f], Z[f]], 1).round(3).tolist()


def crossing(x, lanes, side):
    """edge passes >= 3 m to the inner side of a lane line that it started outside of (diagonal lane-crossing)."""
    for lx in lanes:
        both = np.isfinite(x) & np.isfinite(lx)
        idx = np.flatnonzero(both)
        if len(idx) < 6 or np.median((x[idx[:4]] - lx[idx[:4]]) * side) <= 0:
            continue
        if ((x[both] - lx[both]) * side < -0.5).sum() >= 6:
            return True
    return False


meta = json.load(open(META))["videos"]
rows = []
for rule in ("raw", "iter6", "iter7"):
    os.makedirs(f"{OUT}/{rule}/polylines", exist_ok=True); os.makedirs(f"{OUT}/{rule}/tracks", exist_ok=True)
for clip in CLIPS:
    D = json.load(open(f"{SRC}/polylines/{clip}_polylines.json"))
    fps = float(D.get("fps") or 10.0)
    spd = float(((meta.get(clip) or {}).get("ego_speed") or {}).get("speed_median_mps") or 10.0)
    ds = spd / fps
    for rule in ("raw", "iter6", "iter7"):
        R = copy.deepcopy(D); hold = LM.EdgeHold(); n_cross = n_edge = 0; n_dup = 0
        for f in R["frames"]:
            lanes = [xg(p) for p in f["lanes_bev"]]
            lrec = [{"xg": x} for x in lanes]
            ne, nb = [], []
            for r, p in zip(f["edges"], f["edges_bev"]):
                role = r.get("role") or {-1: "main_L", 1: "main_R"}.get(r.get("side"))
                x = xg(p)
                if role in ("main_L", "main_R") and rule != "raw":
                    sd = -1 if role == "main_L" else 1
                    if rule == "iter6":
                        if not LM.heading_ok(x, max_deg=15.0, lanes=lrec) or LM.crosses_lane(x, lrec):
                            continue
                    else:
                        x = LM.trim_edge(x, sd, lrec)
                        if x is None:
                            continue
                ne.append(dict(r, role=role, xg=x)); nb.append(x)
            if rule == "iter7":
                for role in ("main_L", "main_R"):
                    cur = [e for e in ne if e["role"] == role]
                    h = hold.step(role, dict(cur[0], xg=cur[0]["xg"]) if cur else None, ds)
                    if h is not None and (h.get("hold_filled") or h.get("held")):
                        xt = LM.trim_edge(h["xg"], -1 if role == "main_L" else 1, lrec)
                        h = dict(h, xg=xt) if xt is not None else (cur[0] if cur else None)
                    if cur:
                        i_ = ne.index(cur[0]); ne[i_] = h; nb[i_] = np.asarray(h["xg"])
                    elif h is not None:
                        ne.append(h); nb.append(np.asarray(h["xg"]))
                keepL = LM.suppress_edge_duplicates([{"xg": x, "i": i} for i, x in enumerate(lanes)],
                                                    [e for e in ne if e["role"].startswith("main")])
                ki = {r["i"] for r in keepL}; n_dup += len(lanes) - len(ki)
                f["lanes"] = [r for i, r in enumerate(f["lanes"]) if i in ki]
                f["lanes_bev"] = [p for i, p in enumerate(f["lanes_bev"]) if i in ki]
            for e, x in zip(ne, nb):
                if e["role"] in ("main_L", "main_R"):
                    n_edge += 1; n_cross += crossing(x, lanes, -1 if e["role"] == "main_L" else 1)
            f["edges"] = [{k: v for k, v in e.items() if k != "xg"} for e in ne]
            f["edges_bev"] = [pts(np.asarray(x)) for x in nb]
        json.dump(R, open(f"{OUT}/{rule}/polylines/{clip}_polylines.json", "w"))
        shutil.copy(f"{SRC}/tracks/{clip}_tracks.csv", f"{OUT}/{rule}/tracks/")
        rows.append((clip, rule, n_edge, n_cross, n_dup))
for r in rows:
    print("%-30s %-6s main-edge records %5d  still crossing a lane %4d  lanes suppressed as edge duplicates %4d" % r)
