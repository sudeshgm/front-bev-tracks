#!/usr/bin/env bash
# Anchor3DLane on the front camera. Do not run the Qwen, lift, fuse, or WildDet jobs.
# Do not pip-install torch or mmcv. Kaggle's Python 3.13 cannot build mmcv 1.6.
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("no cuda")
PY

if [[ ! -d third_party/Anchor3DLane/mmseg ]]; then
  rm -rf third_party/Anchor3DLane
  git clone --depth 1 https://github.com/tusen-ai/Anchor3DLane third_party/Anchor3DLane
fi

python - <<'PY'
from pathlib import Path
root = Path("third_party/Anchor3DLane/mmseg/models")
(root / "utils" / "__init__.py").write_text("from .res_layer import ResLayer\n")
(root / "backbones" / "__init__.py").write_text("from .resnet import ResNet, ResNetV1c, ResNetV1d\n")
(root / "losses" / "__init__.py").write_text("from .lane_loss import LaneLoss\n")
(root / "lane_detector" / "__init__.py").write_text(
    "from .anchor_3dlane import Anchor3DLane\nfrom .assigner import TopkAssigner\n"
)
(root / "__init__.py").write_text("from .builder import build_lanedetector\n")
anchor = root / "lane_detector" / "utils" / "anchor.py"
text = anchor.read_text()
text = text.replace("import mmcv\n", "")
text = text.replace(
    "from mmseg.datasets.tools.utils import projection_g2im, resample_laneline_in_y\n",
    "",
)
anchor.write_text(text)
(root / "lane_detector" / "tools.py").write_text(
    "import numpy as np\n\n"
    "def homography_crop_resize(org_img_size, crop_y, resize_img_size):\n"
    "    ratio_x = resize_img_size[1] / org_img_size[1]\n"
    "    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)\n"
    "    return np.array([[ratio_x, 0, 0], [0, ratio_y, -ratio_y * crop_y], [0, 0, 1]])\n"
)
PY

export PYTHONPATH="$(pwd)/compat:$(pwd)/third_party/Anchor3DLane:${PYTHONPATH:-}"

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
