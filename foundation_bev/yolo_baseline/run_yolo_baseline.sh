#!/usr/bin/env bash
# YOLOPv2 baseline dump (measurement only; the foundation pipeline stays YOLO-free).
# Runs the existing front-bev-tracks overlay.py (YOLOPv2 boxes + lanes, offline RTS tracks, whole-clip lane fit)
# with overlay_dump.patch (--dump-dir), then scores it with the same smoothness_metrics.py as the foundation run.
#
# Usage: run_yolo_baseline.sh REPO WEIGHTS DS OUT [DEVICE] [CAMS_JSON] [COMPARE_RUN]
#   REPO        front-bev-tracks checkout dir, or a .tar.gz of it (e.g. sam-bev-22nov ds/fbt_c2b386b.tar.gz)
#   WEIGHTS     YOLOPv2 TorchScript (yolopv2.pt, 1x3x640x640)
#   DS          dir with <clip>.mp4 and motion_assumptions.json
#   OUT         output dir (tracks/, polylines/, run_meta.json, metrics_yolo_vs_<run>.md)
#   DEVICE      unused by overlay.py (it runs YOLOPv2 on CPU via torch.jit, ~0.3-1 s/frame); kept for symmetry
#   CAMS_JSON   cams file with fx fy cx cy height per clip (default DS/cams_iter3.json)
#   COMPARE_RUN foundation run dir to compare against (has tracks/ polylines/ run_meta.json); optional
# Env: CLIPS (default: the 6 KITTI clips), PY (python with torch + opencv; default python3)
set -euo pipefail
REPO=$1; WEIGHTS=$2; DS=$3; OUT=$4; DEVICE=${5:-cpu}; CAMS=${6:-$DS/cams_iter3.json}; CMP=${7:-}
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
export CLIPS=${CLIPS:-"kitti_highway_sample kitti_highway_exit_approach kitti_highway_late kitti_highway_final kitti_city_cyclist_candidate kitti_lane_split_candidate"}
mkdir -p "$OUT/inputs"
OUT=$(cd "$OUT" && pwd)
[ -f "$WEIGHTS" ] || { echo "ERROR: weights not found: $WEIGHTS"; exit 31; }
if [ -f "$REPO" ]; then
  mkdir -p "$OUT/repo" && tar xzf "$REPO" -C "$OUT/repo"; REPO="$OUT/repo"
fi
[ -f "$REPO/scripts/overlay.py" ] || { echo "ERROR: $REPO/scripts/overlay.py missing"; exit 32; }
# iter4: YOLOPv2 head-layout fix for decode_boxes (iter3 IndexError mask [3,80,80] vs [3,1,80,80])
"$PY" "$HERE/decode_fix.py" "$REPO/scripts/overlay.py" || { echo "ERROR: decode_fix did not apply"; exit 34; }
if ! grep -q -- "--dump-dir" "$REPO/scripts/overlay.py"; then
  (cd "$REPO" && patch -p1 < "$HERE/overlay_dump.patch") || { echo "ERROR: overlay_dump.patch did not apply"; exit 33; }
fi
# per-clip motion CSV (t_s, speed_mps, yaw_rate_rps) + CylinderV01 camera JSON from the foundation inputs
"$PY" - "$DS" "$CAMS" "$OUT/inputs" <<'PY'
import json, os, sys, cv2
ds, cams_p, out = sys.argv[1:4]
cams = json.load(open(cams_p)); mot = json.load(open(os.path.join(ds, "motion_assumptions.json")))
if os.path.exists(os.path.join(ds, "motion_private.json")):  # iter13: private clip (fc1_merge), local only
    mot.update(json.load(open(os.path.join(ds, "motion_private.json"))))
meta = {"videos": {}}
for c in os.environ["CLIPS"].split():
    cap = cv2.VideoCapture(os.path.join(ds, f"{c}.mp4")); fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    m = mot.get(c, {}); cam = cams[c]
    sp = cam.get("oxts_speed_mps", m.get("speed_mps", 12.0))
    with open(os.path.join(out, f"{c}_motion.csv"), "w") as f:
        f.write("t_s,speed_mps,yaw_rate_rps\n")
        for k in range(n):
            f.write(f"{k / fps:.4f},{float(sp if not isinstance(sp, list) else sp[min(k, len(sp) - 1)]):.3f},{m.get('yaw_rate_rad_s', 0.0)}\n")
    # pinhole -> CylinderV01 is a close approximation inside the KITTI FOV (u = cx - fx*atan(y/x));
    # the baseline only needs to be scored with the same metric, not to be pixel exact
    json.dump({"fx": cam["fx"], "fy": cam["fy"], "cx": cam["cx"], "cy": cam["cy"], "position_m": [0.0, 0.0, cam.get("height", 1.65)],
               "width": w, "height": h}, open(os.path.join(out, f"{c}_camera.json"), "w"))
    meta["videos"][c] = {"fps": fps, "ego_speed_mps": sp if not isinstance(sp, list) else None, "frames": n}
json.dump(meta, open(os.path.join(os.path.dirname(out), "run_meta.json"), "w"), indent=1)
print("inputs ok", list(meta["videos"]))
PY
rc_all=0
for c in $CLIPS; do
  echo "== YOLO $c"
  set +e
  (cd "$REPO" && "$PY" scripts/overlay.py --video "$DS/$c.mp4" --motion "$OUT/inputs/${c}_motion.csv" \
      --camera "$OUT/inputs/${c}_camera.json" --weights "$WEIGHTS" --lane-backend yolopv2 \
      --cache "$OUT/inputs/cache_$c.npz" --output "$OUT/${c}_overlay.mp4" --dump-dir "$OUT") > "$OUT/log_$c.txt" 2>&1
  rc=$?
  set -e
  echo "   rc=$rc"; [ $rc -eq 0 ] || { rc_all=$rc; tail -5 "$OUT/log_$c.txt"; }
done
MET="$HERE/../src/smoothness_metrics.py"; [ -f "$MET" ] || MET="$DS/src/smoothness_metrics.py"; [ -f "$MET" ] || MET="$(dirname "$HERE")/smoothness_metrics.py"
if [ -n "$CMP" ] && [ -d "$CMP/tracks" ]; then
  "$PY" "$MET" --run "$OUT" --label yolopv2_baseline --run "$CMP" --label "$(basename "$CMP")" \
      --out "$OUT/metrics_yolo_vs_run.md" --json "$OUT/metrics_yolo_vs_run.json"
else
  "$PY" "$MET" --run "$OUT" --label yolopv2_baseline --out "$OUT/metrics_yolo.md" --json "$OUT/metrics_yolo.json"
fi
[ $rc_all -eq 0 ] && echo "YOLO BASELINE OK" || { echo "YOLO BASELINE FAILED rc=$rc_all"; exit $rc_all; }
