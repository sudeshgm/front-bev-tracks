"""iter12: compare runs (GPU or replay) on the iter12 targets.
usage: cmp_runs.py LABEL=RUN_DIR [LABEL=RUN_DIR ...] [--yolo metrics_yolo_vs_run.json] [--md out.md] [--json out.json]
Each RUN_DIR has metrics.json (smoothness_metrics.py) and polylines/. Prints per clip: accel, idsw, lane / edge pres,
lane / edge wjit, lane / edge jit2, gore frames, munich R z10 frame diff vs the first run, frame-0 completeness."""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from analyze_polylines import side_series, x_at  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--yolo"); ap.add_argument("--md"); ap.add_argument("--json")
a = ap.parse_args()
SH = {"kitti_highway_sample": "sample", "kitti_highway_exit_approach": "exit", "kitti_highway_late": "late",
      "kitti_highway_final": "final", "kitti_city_cyclist_candidate": "city", "kitti_lane_split_candidate": "lane_split",
      "munich": "munich", "a2d2a": "a2d2a", "a2d2b": "a2d2b"}
def load(run):
    m = json.load(open(f"{run}/metrics.json")); m = next(iter(m.values())) if isinstance(m, dict) else m
    return {r["clip"]: r for r in m}
def frames_info(run, clip):
    d = json.load(open(f"{run}/polylines/{clip}_polylines.json")); F = d["frames"]
    out = {"gores": [f["frame"] for f in F if f.get("gores")]}
    for kind in ("lanes", "edges"):  # live-only width jitter: frames whose nearest L / R line is backfilled or
        hw = []                         # anchored (not measured on that frame) are left out
        for f in F:
            L = R = None
            for r, p in zip(f[kind], f[kind + "_bev"]):
                v = x_at(p, 10.0)
                if v is None: continue
                if v < 0 and (L is None or v > L[0]): L = (v, r)
                if v >= 0 and (R is None or v < R[0]): R = (v, r)
            ok = L is not None and R is not None and not any(q[1].get("backfill") or q[1].get("anchored") for q in (L, R))
            hw.append((R[0] - L[0]) / 2 if ok else None)
        d = [abs(b - a) for a, b in zip(hw, hw[1:]) if a is not None and b is not None]
        out[f"{kind}_wjit_live"] = round(float(np.median(d)), 3) if d else None
    for kind in ("lanes", "edges"):
        Ls, Rs = side_series(F, kind + "_bev", 10.0)
        out[f"{kind}_L"] = np.array([v is not None for v in Ls]); out[f"{kind}_R"] = np.array([v is not None for v in Rs])
    return out
runs = [r.split("=", 1) for r in a.runs]
M = {lab: load(run) for lab, run in runs}
FI = {lab: {c: frames_info(run, c) for c in SH if os.path.exists(f"{run}/polylines/{c}_polylines.json")} for lab, run in runs}
Y = {}
if a.yolo:
    y = json.load(open(a.yolo)); y = y.get("yolopv2_baseline", y) if isinstance(y, dict) else y
    if isinstance(y, dict) and "clip" not in y: y = list(y.values()) if all(isinstance(v, dict) for v in y.values()) else [y]
    Y = {r["clip"]: r for r in y if isinstance(r, dict) and "clip" in r}
def f3(v): return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) and abs(v) < 10 else str(v))
labs = [l for l, _ in runs]
L = []
L.append("| clip | metric | " + " | ".join(labs) + (" | YOLO |" if Y else " |"))
L.append("|---|---|" + "---|" * (len(labs) + (1 if Y else 0)))
KEYS = [("accel", "accel_rms_mps2"), ("idsw", "id_switch_proxy_per100f"), ("lane pres L/R", ("lanes_L10_pres", "lanes_R10_pres")),
        ("lane wjit", "lanes_10_wjit"), ("lane jit2 L/R", ("lanes_L10_jit2", "lanes_R10_jit2")),
        ("edge pres L/R", ("edges_L10_pres", "edges_R10_pres")), ("edge wjit", "edges_10_wjit"),
        ("edge jit2 L/R", ("edges_L10_jit2", "edges_R10_jit2"))]
res = {}
for c, sh in SH.items():
    if not all(c in M[l] for l in labs): continue
    for nm, k in KEYS:
        def val(r):
            if r is None: return "—"
            if isinstance(k, tuple): return "/".join(f3(r.get(x)) for x in k)
            return f3(r.get(k))
        row = [val(M[l][c]) for l in labs]
        if Y: row.append(val(Y.get(c)) if nm.startswith("lane") else "")
        L.append(f"| {sh} | {nm} | " + " | ".join(row) + " |")
        res.setdefault(sh, {})[nm] = row
    for kind, nm in (("lanes", "lane wjit live"), ("edges", "edge wjit live")):
        row = [f3(FI[l][c].get(f"{kind}_wjit_live")) for l in labs]
        L.append(f"| {sh} | {nm} | " + " | ".join(row) + (" | |" if Y else " |")); res[sh][nm] = row
    g = [FI[l][c]["gores"] for l in labs]
    def rng(fs):
        if not fs: return "—"
        o, st = [], fs[0]
        for p, q in zip(fs, fs[1:] + [None]):
            if q != p + 1: o.append(f"{st}-{p}" if p != st else f"{st}"); st = q
        return ",".join(o)
    L.append(f"| {sh} | gores | " + " | ".join(rng(x) for x in g) + (" | |" if Y else " |"))
    f0 = []
    for l in labs:
        fi = FI[l][c]; f0.append("".join(("✓" if fi.get(k) is not None and len(fi[k]) and fi[k][0] else "✗")
                                         for k in ("lanes_L", "lanes_R", "edges_L", "edges_R")))
    L.append(f"| {sh} | f0 laneL,R edgeL,R | " + " | ".join(f0) + (" | |" if Y else " |"))
    res[sh]["gores"] = [rng(x) for x in g]; res[sh]["f0"] = f0
b = labs[0]
for l in labs[1:]:
    for c in ("munich",):
        if c not in FI[b] or c not in FI[l]: continue
        x0, x1 = FI[b][c]["edges_R"], FI[l][c]["edges_R"]; n = min(len(x0), len(x1))
        gain = np.flatnonzero(x1[:n] & ~x0[:n]).tolist(); lost = np.flatnonzero(x0[:n] & ~x1[:n]).tolist()
        L.append(f"\nmunich R z10 {b} {int(x0.sum())} -> {l} {int(x1.sum())}: gained {gain}  lost {lost}")
        res[f"munich_R_{l}"] = {"n": int(x1.sum()), "gain": gain, "lost": lost}
txt = "\n".join(L); print(txt)
if a.md: open(a.md, "w").write(txt + "\n")
if a.json: json.dump(res, open(a.json, "w"))
