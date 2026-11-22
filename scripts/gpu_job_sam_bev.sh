#!/usr/bin/env bash
# SAM + Depth Anything lane/road-edge BEV overlay. Set VIDEO.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${VIDEO:-}" ]]; then
  echo "Set VIDEO" >&2
  exit 2
fi

python -m pip install -q --upgrade "setuptools<80"
python -m pip install -q opencv-python-headless huggingface_hub
# Depth Anything V2 Small
if [[ ! -f weights/depth_anything_v2_vits.pth ]]; then
  python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth", local_dir="weights")
print("depth checkpoint ready")
PY
fi
# SAM 2.1 Small (optional; script falls back to edge proxy)
if [[ ! -f weights/sam2.1_hiera_small.pt ]]; then
  python - <<'PY'
from huggingface_hub import hf_hub_download
try:
    hf_hub_download("facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt", local_dir="weights")
    print("sam2 ready")
except Exception as e:
    print("sam2 download skipped", e)
PY
fi

mkdir -p results
python scripts/model_compare/sam_depth_bev.py \
  --video "$VIDEO" \
  --output "results/$(basename "$VIDEO" .mp4)_sam_bev.mp4" \
  --fx "${FX:-800}" --fy "${FY:-800}" --cx "${CX:-640}" --cy "${CY:-360}" \
  --device cuda:0
echo "done $(basename "$VIDEO" .mp4)_sam_bev.mp4"
