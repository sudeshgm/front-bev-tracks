#!/bin/bash
# iter14 replay of the iter13 GPU run (post-tracker only; stage-2 changes ego14 / night gate need a stage-2 run):
#   rep13 = iter14 src with iter13 behaviour, iter14 = defaults, bf_trim = FBEV_EDGE_CAR_BF=trim, gore_spatial = FBEV_GORE_SPATIAL=1
set -eu
H=/workspace/foundation_bev; S=$H/iter14_spec; RUN=${RUN:-$H/iter13}; O=${1:-$S/replay}
mkdir -p $O
FBEV_EDGE_CAR_BF=push python3 $S/tools/replay_full.py --run $RUN --out $O/rep13 --held auto --json $O/gore_rep13.json > $O/gore_rep13.txt
python3 $S/tools/replay_full.py --run $RUN --out $O/iter14 --held auto --json $O/gore_iter14.json > $O/gore_iter14.txt
FBEV_EDGE_CAR_BF=trim python3 $S/tools/replay_full.py --run $RUN --out $O/bf --held auto --json $O/gore_bf.json > $O/gore_bf.txt
FBEV_GORE_SPATIAL=1 python3 $S/tools/replay_full.py --run $RUN --out $O/sp --held auto --json $O/gore_sp.json > $O/gore_sp.txt
for n in rep13 iter14 bf sp; do python3 $S/src/smoothness_metrics.py --run $O/$n --label $n --json $O/$n/metrics.json --out $O/$n/metrics.md > /dev/null; done
python3 $S/tools/cmp_runs.py gpu13=$RUN rep13=$O/rep13 iter14=$O/iter14 bf_trim=$O/bf gore_spatial=$O/sp --yolo $RUN/metrics_yolo_vs_run.json --md $O/replay_vs_iter13.md --json $O/replay_vs_iter13.json
for n in rep13 iter14 bf; do python3 $S/tools/car_audit.py --run $O/$n --dets $RUN/dets_dump --json $O/car_audit_$n.json > $O/car_audit_$n.txt; done
