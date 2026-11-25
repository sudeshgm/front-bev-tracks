#!/bin/bash
# iter15 GPU run vs the iter13 GPU run (iter14 = iter13 public output; no iter14 GPU run): metrics table, car audit,
# night-gate counts, ego sources + lane pitch, and the ego error vs KITTI OXTS (public GT) -> $NEW/metrics_vs_iter13.md
set -eu
S=$(cd "$(dirname "$0")/.." && pwd); OLD=${OLD:-/workspace/foundation_bev/iter13}; NEW=${NEW:-/workspace/foundation_bev/iter15}
python3 $S/tools/cmp_runs.py gpu13=$OLD gpu15=$NEW --yolo $NEW/metrics_yolo_vs_run.json --md $NEW/cmp_vs_iter13.md --json $NEW/cmp_vs_iter13.json
python3 $S/tools/car_audit.py --run $NEW --dets $NEW/dets_dump --json $NEW/car_audit.json > $NEW/car_audit.txt
python3 $S/tools/diff_scope.py $OLD $NEW kitti_highway_sample kitti_highway_exit_approach kitti_highway_late kitti_highway_final \
  kitti_city_cyclist_candidate kitti_lane_split_candidate munich a2d2a a2d2b > $NEW/diff_scope_vs_iter13.txt
A=""; for c in kitti_highway_sample kitti_highway_exit_approach kitti_highway_late kitti_highway_final kitti_city_cyclist_candidate kitti_lane_split_candidate; do
  A="$A $c:$OLD:$NEW:sub:$S/kitti_oxts_subset.json:$c"; done
python3 $S/tools/ego_eval.py $NEW/ego_eval_oxts.json $A > $NEW/ego_eval_oxts.txt
{ echo "# iter15 GPU vs iter13 GPU"; echo; cat $NEW/cmp_vs_iter13.md; echo; echo "## Night gate (TB_ADAPT / VPLANE auto): gate-on counts per clip (must be 0)"; echo
  python3 $S/tools/gate_summary.py $NEW/gate.jsonl; echo; echo "## Car audit"; echo; echo '```'; cat $NEW/car_audit.txt; echo '```'
  echo; echo "## Change scope vs iter13 (tracks / ego pose / lines)"; echo; echo '```'; cat $NEW/diff_scope_vs_iter13.txt; echo '```'
  echo; echo "## Ego vs KITTI OXTS (before = iter13 GPU, signfix = iter13 psi negated, after = iter15 GPU)"; echo; echo '```'; cat $NEW/ego_eval_oxts.txt; echo '```'
  echo; echo "## Ego speed sources + lane pitch"; echo; python3 - "$NEW" <<'PY'
import json, sys, glob, os
for f in sorted(glob.glob(os.path.join(sys.argv[1], "results", "*.json"))):
    try:
        e = json.load(open(f)).get("ego_speed") or {}
    except Exception:
        continue
    if e:
        print(f"- {os.path.basename(f)[:-5]}: {e.get('source')} {e.get('speed_median_mps', 0):.2f} m/s; yaw {e.get('yaw_source')}; lane pitch {e.get('ego_pitch')}")
PY
} > $NEW/metrics_vs_iter13.md
echo "wrote $NEW/metrics_vs_iter13.md"
