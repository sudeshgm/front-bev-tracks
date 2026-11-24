#!/bin/bash
# iter10 full replay on saved GPU runs (CPU only). usage: replay_iter10.sh ITER8_RUN ITER9_RUN WORK
# ITER9_RUN needs polylines/ (with ego + edges_raw), tracks/, dets_dump/*_dets_ego.json; ITER8_RUN polylines/ (unsmoothed).
set -eu
I8=$1; I9=$2; W=$3; T=$(cd "$(dirname "$0")" && pwd); S=$T/..
mkdir -p $W/egorun/polylines
for f in $I9/polylines/*_polylines.json $I9/dets_dump/*_dets_ego.json; do ln -sf $f $W/egorun/polylines/; done
lanes() { N=$1; shift; rm -rf $W/rl/$N; python3 $T/replay_lanes.py $I8 $I9 $W/rl/$N "$@" >/dev/null; }
edges() { N=$1; shift; rm -rf $W/rk/$N; mkdir -p $W/rk/$N; python3 $T/replay_smooth.py $I8 $W/egorun $W/rk/$N --kerb --kinds edges "$@" > $W/rk/$N/log.txt; }
lanes base --pair 0 --backfill 0
lanes iter10 --pair 1 --backfill 1
FBEV_KERB_RAMP_S=0 edges base
edges iter10
for v in base iter10; do
  python3 - "$W" "$v" <<'P'
import json, glob, os, sys
W, v = sys.argv[1], sys.argv[2]; o = f"{W}/comb_{v}"
os.makedirs(o + "/polylines", exist_ok=True); os.makedirs(o + "/tracks", exist_ok=True)
for p in glob.glob(f"{W}/rk/{v}/polylines/*_polylines.json"):
    c = os.path.basename(p); d = json.load(open(p)); L = json.load(open(f"{W}/rl/{v}/polylines/{c}"))
    for F, G in zip(d["frames"], L["frames"]):
        F["lanes"], F["lanes_bev"] = G["lanes"], G["lanes_bev"]
    json.dump(d, open(f"{o}/polylines/{c}", "w"))
    t = os.path.realpath(f"{W}/rk/{v}/tracks/" + c.replace("_polylines.json", "_tracks.csv")); dst = f"{o}/tracks/" + os.path.basename(t)
    if os.path.exists(t) and not os.path.exists(dst): os.symlink(t, dst)
P
  python3 $S/src/smoothness_metrics.py --run $W/comb_$v --label $v --json $W/comb_$v/metrics.json --out $W/comb_$v/metrics.md >/dev/null
done
python3 $T/replay_table10.py $W/comb_base/metrics.json $W/comb_iter10/metrics.json $I9/metrics_yolo_vs_run.json $W/replay_vs_iter9
RUN=$I9 OUT=$W/replay_gap.json python3 $T/replay_gap.py > $W/replay_gap.txt
echo "wrote $W/replay_vs_iter9.md $W/replay_gap.txt"
