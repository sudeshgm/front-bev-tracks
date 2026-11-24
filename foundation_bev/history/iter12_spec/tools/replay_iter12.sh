#!/bin/bash
# iter12 replay: the iter11 GPU run (step + edges_raw) through the full post-tracker pipeline with
#   rep11  = iter11_spec/src (fidelity check vs the GPU output)
#   iter12 = iter12_spec/src defaults
# then cmp_runs vs gpu11 and the YOLOPv2 baseline.   usage: tools/replay_iter12.sh [OUT]
set -eu
H=/workspace/foundation_bev; S=$H/iter12_spec; RUN=${RUN:-$H/iter11}; O=${1:-$H/iter12_work/replay}
mkdir -p $O
FBEV_REPLAY_SRC=$H/iter11_spec/src python3 $S/tools/replay_full.py --run $RUN --out $O/rep11 --json $O/gore_rep11.json > $O/gore_rep11.txt
python3 $S/tools/replay_full.py --run $RUN --out $O/iter12 --json $O/gore_iter12.json > $O/gore_iter12.txt
for n in rep11 iter12; do python3 $S/src/smoothness_metrics.py --run $O/$n --label $n --json $O/$n/metrics.json --out $O/$n/metrics.md > /dev/null; done
python3 $S/tools/cmp_runs.py gpu11=$RUN rep11=$O/rep11 iter12=$O/iter12 --yolo $RUN/metrics_yolo_vs_run.json --md $O/replay_vs_iter11.md --json $O/replay_vs_iter11.json
