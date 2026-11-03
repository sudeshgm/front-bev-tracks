#!/usr/bin/env bash
# WildDet3D on one front-camera video. Set VIDEO and CAMERA. Does not fetch a dataset.
# Do not pip-install torch. A T4 has 16 GB; the checkpoint is 4.7 GB.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${VIDEO:-}" || -z "${CAMERA:-}" ]]; then
  echo "Set VIDEO and CAMERA to local files." >&2
  exit 2
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("no cuda")
PY

python -m pip install -q --upgrade "setuptools<80"
python -m pip install -q --no-deps einops timm "ftfy==6.1.1" iopath portalocker regex huggingface_hub pyquaternion scipy ml_collections
python -m pip install -q --no-deps "utils3d @ git+https://github.com/EasternJournalist/utils3d.git"
python -c "import vis4d" 2>/dev/null || python -m pip install -q vis4d==1.0.0
python -c "import vis4d_cuda_ops" 2>/dev/null || python -m pip install -q "git+https://github.com/SysCV/vis4d_cuda_ops.git" --no-build-isolation --no-cache-dir

if [[ ! -d third_party/WildDet3D/wilddet3d ]]; then
  rm -rf third_party/WildDet3D
  git clone --depth 1 --recurse-submodules https://github.com/allenai/WildDet3D third_party/WildDet3D
fi
if [[ ! -d third_party/WildDet3D/third_party/sam3 ]]; then
  git -C third_party/WildDet3D submodule update --init --recursive
fi

mkdir -p weights results
if [[ ! -f weights/wilddet3d_alldata_all_prompt_v1.0.pt ]]; then
  python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    "allenai/WildDet3D",
    "wilddet3d_alldata_all_prompt_v1.0.pt",
    local_dir="weights",
)
print("checkpoint", path, flush=True)
PY
fi

python scripts/wilddet_clip.py \
  --video "$VIDEO" \
  --camera "$CAMERA" \
  --repeat "${REPEAT:-1}" \
  --output results/wilddet.json \
  --device cuda:0
echo "wilddet clip done. Return results/wilddet.json and results/wilddet.md"
