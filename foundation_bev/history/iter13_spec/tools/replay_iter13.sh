#!/bin/bash
# iter13 replay: the iter12 GPU run (step + edges_raw + edges_raw_held) through the full post-tracker pipeline with
#   rep12  = iter12_spec/src (fidelity check vs the GPU output)
#   iter13 = iter13_spec/src defaults
# then cmp_runs vs gpu12 and the YOLOPv2 baseline, and the parked-car audit of every output main edge.
# usage: tools/replay_iter13.sh [OUT]
set -eu
H=/workspace/foundation_bev; S=$H/iter13_spec; RUN=${RUN:-$H/iter12}; O=${1:-$H/iter13_work/replay}
mkdir -p $O
FBEV_REPLAY_SRC=$H/iter12_spec/src python3 $S/tools/replay_full.py --run $RUN --out $O/rep12 --held auto --json $O/gore_rep12.json > $O/gore_rep12.txt
python3 $S/tools/replay_full.py --run $RUN --out $O/iter13 --held auto --json $O/gore_iter13.json > $O/gore_iter13.txt
for n in rep12 iter13; do python3 $S/src/smoothness_metrics.py --run $O/$n --label $n --json $O/$n/metrics.json --out $O/$n/metrics.md > /dev/null; done
python3 $S/tools/cmp_runs.py gpu12=$RUN rep12=$O/rep12 iter13=$O/iter13 --yolo $RUN/metrics_yolo_vs_run.json --md $O/replay_vs_iter12.md --json $O/replay_vs_iter12.json
for n in gpu12 rep12 iter13; do d=$O/$n; [ $n = gpu12 ] && d=$RUN; python3 $S/tools/car_audit.py --run $d --dets $RUN/dets_dump --json $O/car_audit_$n.json > $O/car_audit_$n.txt; done
