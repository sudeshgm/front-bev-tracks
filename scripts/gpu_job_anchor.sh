#!/usr/bin/env bash
# Anchor3DLane on the front camera. Do not run the Qwen, lift, fuse, or WildDet jobs.
# Do not pip-install torch. mmcv is imported, not compiled with CUDA ops.
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("no cuda")
PY

if ! python -c "import mmcv" >/dev/null 2>&1; then
  MMCV_WITH_OPS=0 python -m pip install -q "mmcv==1.6.2"
fi
python -c "import mmcv; assert mmcv.__version__.startswith('1.6'), mmcv.__version__"

if [[ ! -d third_party/Anchor3DLane/mmseg ]]; then
  rm -rf third_party/Anchor3DLane
  git clone --depth 1 https://github.com/tusen-ai/Anchor3DLane third_party/Anchor3DLane
fi
export PYTHONPATH="$(pwd)/third_party/Anchor3DLane:${PYTHONPATH:-}"

mkdir -p weights
if [[ ! -f weights/openlane_anchor3dlane.pth ]]; then
  python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    "nowherespyfly/anchor3dlane",
    "Openlane/openlane_anchor3dlane.pth",
    local_dir="weights",
)
print("checkpoint", path, flush=True)
PY
  mv -f weights/Openlane/openlane_anchor3dlane.pth weights/openlane_anchor3dlane.pth
fi

expected=115
have=$(find data/frames -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [[ "$have" -ne "$expected" ]]; then
  rm -rf data/frames
  python scripts/fetch_clip.py --scene gaimersheim --start 12025 --stride 1 --count 115 \
    --pack-dir data/frames --cache "${CACHE:-$PWD/data/cache}"
fi

mkdir -p results
python scripts/anchor3d.py \
  --frames data/frames \
  --output results \
  --checkpoint weights/openlane_anchor3dlane.pth \
  --config third_party/Anchor3DLane/configs/openlane/anchor3dlane.py \
  --device cuda:0
echo "anchor3d job done. Return results/eval.json and results/summary.md"
