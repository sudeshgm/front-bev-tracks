#!/bin/bash
# iter14 GPU run vs the iter13 GPU run: metrics table, car audit, ego table, night-gate counts -> $NEW/metrics_vs_iter13.md
set -eu
S=$(cd "$(dirname "$0")/.." && pwd); OLD=${OLD:-/workspace/foundation_bev/iter13}; NEW=${NEW:-/workspace/foundation_bev/iter14}
python3 $S/tools/cmp_runs.py gpu13=$OLD gpu14=$NEW --yolo $NEW/metrics_yolo_vs_run.json --md $NEW/cmp_vs_iter13.md --json $NEW/cmp_vs_iter13.json
python3 $S/tools/car_audit.py --run $NEW --dets $NEW/dets_dump --json $NEW/car_audit.json > $NEW/car_audit.txt
{ echo "# iter14 GPU vs iter13 GPU"; echo; cat $NEW/cmp_vs_iter13.md; echo; echo "## Night gate (TB_ADAPT / VPLANE auto): gate-on counts per clip (must be 0)"; echo
  python3 $S/tools/gate_summary.py $NEW/gate.jsonl; echo; echo "## Car audit"; echo; echo '```'; cat $NEW/car_audit.txt; echo '```'
  echo; echo "## Ego speed sources"; echo; python3 - "$NEW" <<'PY'
import json, sys, glob, os
for f in sorted(glob.glob(os.path.join(sys.argv[1], "results", "*.json"))):
    try:
        e = json.load(open(f)).get("ego_speed") or {}
    except Exception:
        continue
    if e:
        print(f"- {os.path.basename(f)[:-5]}: {e.get('source')} {e.get('speed_median_mps', 0):.2f} m/s (bev_road {e.get('bev_road_mps')}, valid {e.get('bev_road_valid')})")
PY
} > $NEW/metrics_vs_iter13.md
echo "wrote $NEW/metrics_vs_iter13.md"
