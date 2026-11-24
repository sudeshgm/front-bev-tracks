#!/bin/bash
# iter11 full replay on saved GPU runs (CPU only). usage: replay_iter11.sh ITER8_RUN ITER9_RUN ITER10_RUN WORK
#  ITER8_RUN: polylines/ (unsmoothed lanes); ITER9_RUN: polylines/ (lanes for the gore replay, kerb oracle) + dets_dump/;
#  ITER10_RUN: polylines/ (edges_raw, ego, GPU gores), tracks/, dets_dump/, metrics_yolo_vs_run.json.
# 1) gore gate: the iter10 settings must reproduce the iter10 GPU widen gores (exit f21-26 AND munich f881-885)
#    before the iter11 result counts; 2) lanes; 3) edges + metrics; 4) table.
set -eu
I8=$1; I9=$2; I10=$3; W=$4; T=$(cd "$(dirname "$0")" && pwd); S=$T/..
CL=kitti_lane_split_candidate,kitti_highway_exit_approach,kitti_highway_sample,kitti_highway_late,kitti_highway_final,kitti_city_cyclist_candidate,munich,a2d2a,a2d2b
mkdir -p $W
G="--lanes-run $I9 --kerb-oracle $I9 --clips $CL"
python3 $T/replay_gore.py --run $I10 $G --set iter10 --json $W/gore_set10.json > $W/gore_set10.txt
python3 $T/replay_gore.py --run $I9 $G --set iter9 --json $W/gore_set9.json > $W/gore_set9.txt
python3 - "$W" <<'P'
import json, sys
W = sys.argv[1]; d = json.load(open(f"{W}/gore_set10.json")); d9 = json.load(open(f"{W}/gore_set9.json"))
rep = lambda c, D: sorted({f for f, s in D[c]["replay"] if "widen" in s})
gpu = lambda c, D: sorted({f for f, s in D[c]["gpu"] if "widen" in s})
ok = all(rep(c, d) == gpu(c, d) for c in d) and all(rep(c, d9) == gpu(c, d9) for c in d9) \
     and set(range(881, 886)) <= set(rep("munich", d)) and set(range(21, 27)) <= set(rep("kitti_highway_exit_approach", d))
print("GORE REPLAY TRUSTWORTHY" if ok else "GORE REPLAY DOES NOT REPRODUCE THE GPU WIDEN GORES", flush=True)
sys.exit(0 if ok else 3)
P
lanes() { N=$1; shift; rm -rf $W/rl/$N; python3 $T/replay_lanes.py $I8 $I9 $W/rl/$N --clips $CL "$@" >/dev/null; }
FBEV_PAIR_W_MODE=tri FBEV_LANE_BACKFILL_S=0.3 lanes iter10
lanes iter11
rm -rf $W/full10 $W/full11
python3 $T/replay_gore.py --run $I10 $G --set iter10 --out $W/full10 --lanes-poly $W/rl/iter10 > /dev/null
python3 $T/replay_gore.py --run $I10 $G --out $W/full11 --lanes-poly $W/rl/iter11 --json $W/gore_iter11.json > $W/gore_iter11.txt
for v in full10 full11; do python3 $S/src/smoothness_metrics.py --run $W/$v --label $v --json $W/$v/metrics.json --out $W/$v/metrics.md >/dev/null; done
python3 $T/replay_table11.py $W/full10/metrics.json $W/full11/metrics.json $I10/metrics_yolo_vs_run.json $W/replay_vs_iter10
grep -v "^   gpu" $W/gore_iter11.txt
