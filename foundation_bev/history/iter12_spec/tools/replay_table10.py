"""iter10 replay table: YOLOPv2 (baseline only) | iter9 GPU | iter9 replay | iter10 replay | delta (replay vs replay).
iter9 replay = iter8 GPU lanes + centre smoothing 0.3 s (replay_lanes.py --pair 0 --backfill 0) and iter8 GPU edges +
KerbState (FBEV_KERB_RAMP_S=0) + smoothing (replay_smooth.py --kerb --kinds edges); iter10 replay = same with
pair_width_smooth + lane_backfill and the 0.3 s kerb ramp. Edge gap bridging (target a) needs the pipeline (trim runs
before hold / kerb): see replay_gap.txt for its z = 10 m presence estimate.
usage: replay_table10.py BASE_METRICS COMB_METRICS GPU_YOLO_JSON OUT_PREFIX"""
import json, sys
def load(p):
    d = json.load(open(p))
    if isinstance(d, dict) and len(d) == 1:
        d = list(d.values())[0]
    return {r["clip"]: r for r in d}
B, C = load(sys.argv[1]), load(sys.argv[2])
G = json.load(open(sys.argv[3])); Y = {r["clip"]: r for r in G["yolopv2_baseline"]}; W = {r["clip"]: r for r in G["working"]}
keys = ["lanes_10_wjit", "lanes_L10_jit2", "lanes_R10_jit2", "lanes_L10_pres", "lanes_R10_pres", "lanes_12_wjit",
        "edges_10_wjit", "edges_L10_jit2", "edges_R10_jit2", "edges_L10_pres", "edges_R10_pres",
        "accel_rms_mps2", "id_switch_proxy_per100f"]
clips = [c for c in W if c in C]
out = {}; md = ["# iter10 replay vs iter9 (and YOLOPv2)", "",
                "Replay on the iter9 / iter8 GPU outputs (CPU, no SAM re-run). `d` = iter10 replay - iter9 replay. "
                "YOLOPv2 is a measurement baseline only (not in the pipeline).", ""]
for k in keys:
    md += [f"## {k}", "", "| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |", "|---|---|---|---|---|---|"]
    for c in clips:
        y, g, b, n = Y.get(c, {}).get(k), W[c].get(k), B.get(c, {}).get(k), C[c].get(k)
        d = None if (b is None or n is None) else round(n - b, 3)
        out.setdefault(c, {})[k] = {"yolo": y, "iter9_gpu": g, "iter9_replay": b, "iter10_replay": n, "d": d}
        f = lambda v: "" if v is None else str(v)
        md.append(f"| {c} | {f(y)} | {f(g)} | {f(b)} | {f(n)} | {'' if d is None else ('%+.3f' % d)} |")
    md.append("")
open(sys.argv[4] + ".md", "w").write("\n".join(md)); json.dump(out, open(sys.argv[4] + ".json", "w"), indent=1)
