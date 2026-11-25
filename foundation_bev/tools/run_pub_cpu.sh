#!/bin/bash
# iter14 public check with the REAL stage-2 models on CPU (SAM 2.1 hiera-s + DAv2-S, iter12 GPU WD3D dets).
# Usage: run_pub_real.sh SRC_DIR OUT_DIR ; per-clip outputs written as each clip finishes (resumable: skips done clips).
H=/workspace/foundation_bev; SRC=$1; O=$2; DS=/workspace/gpubot/foundation-bev-23nov/ds; V=/workspace/fbev_cpu
mkdir -p $O/out $O/stills $O/tracks $O/polylines $O/results $O/logs
export PYTHONPATH="$V/dav2:${PYTHONPATH:-}" OMP_NUM_THREADS=8
cd $SRC
for tag in ${CLIPS:-kitti_highway_sample kitti_highway_exit_approach kitti_lane_split_candidate a2d2a}; do
  [ -f $O/results/$tag.json ] && { echo "skip $tag"; continue; }
  spd=$($V/venv/bin/python -c "import json;print(json.load(open('$H/iter4/run_meta.json'))['videos']['$tag']['ego_speed']['prior_mps'])")
  $V/venv/bin/python foundation_bev.py --video $DS/$tag.mp4 --output $O/out/${tag}_iter14.mp4 \
    --cam-json $H/iter14_work/cams_iter13.json --require-dets 1 --tag $tag --speed $spd --device cpu --weights $V/weights \
    --stills $O/stills --tracks-csv $O/tracks/${tag}_tracks.csv --polylines-json $O/polylines/${tag}_polylines.json \
    --clip-meta $O/results/$tag.json --dets $H/iter12/dets/wd3d_${tag}.npz > $O/logs/$tag.txt 2>&1
  echo "$tag rc=$? $(date +%H:%M) $(grep '^EGO' $O/logs/$tag.txt | tail -1)"
done
