"""iter11 replay table: YOLOPv2 (measurement baseline only) | iter10 GPU | iter10 replay | iter11 replay | d (replay vs replay).
usage: replay_table11.py ITER10_REPLAY_METRICS ITER11_REPLAY_METRICS ITER10_GPU_YOLO_JSON OUT_PREFIX"""
import json, sys
def load(p):
    d = json.load(open(p))
    if isinstance(d, dict) and len(d) == 1:
        d = list(d.values())[0]
    return {r["clip"]: r for r in d}
B, C = load(sys.argv[1]), load(sys.argv[2])
G = json.load(open(sys.argv[3])); Y = {r["clip"]: r for r in G["yolopv2_baseline"]}; W = {r["clip"]: r for r in G["working"]}
keys = ["lanes_10_wjit", "lanes_12_wjit", "lanes_L10_jit2", "lanes_R10_jit2", "lanes_L10_pres", "lanes_R10_pres",
        "edges_10_wjit", "edges_L10_jit2", "edges_R10_jit2", "edges_L10_pres", "edges_R10_pres",
        "accel_rms_mps2", "n_tracks", "id_switch_proxy_per100f"]
clips = sorted(c for c in W if c in C)
out = {}; md = ["# iter11 replay vs iter10 (and YOLOPv2)", "",
                "CPU replay on the iter10 GPU outputs (no SAM re-run): edges = iter10 GPU `edges_raw` through the "
                "pipeline's own `StageLines.post_edges` (trim, kerb, hold, veto, gores) + centre smoothing + edge "
                "backfill; lanes = iter8 GPU unsmoothed lanes + centre smoothing + pair width + lane backfill. "
                "`iter10 replay` = the same harness with the iter11 switches off; it reproduces the iter10 GPU numbers "
                "(lanes exactly; edges within 0.002 except lane_split R jit2). `d` = iter11 replay - iter10 replay. "
                "YOLOPv2 is a measurement baseline only (not in the pipeline).", ""]
for k in keys:
    md += [f"## {k}", "", "| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |", "|---|---|---|---|---|---|"]
    for c in clips:
        y, g, b, n = Y.get(c, {}).get(k), W[c].get(k), B.get(c, {}).get(k), C[c].get(k)
        d = None if (b is None or n is None) else round(n - b, 3)
        out.setdefault(c, {})[k] = {"yolo": y, "iter10_gpu": g, "iter10_replay": b, "iter11_replay": n, "d": d}
        f = lambda v: "" if v is None else str(v)
        md.append(f"| {c} | {f(y)} | {f(g)} | {f(b)} | {f(n)} | {'' if d is None else ('%+.3f' % d)} |")
    md.append("")
json.dump(out, open(sys.argv[4] + ".json", "w"), indent=1)
open(sys.argv[4] + ".md", "w").write("\n".join(md) + "\n")
print("wrote", sys.argv[4] + ".md")
