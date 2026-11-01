#!/usr/bin/env bash
# Depth-agreement pass. Do not run scripts/gpu_job.sh or scripts/gpu_job_lift.sh.
# Kaggle T4. Do not pip-install torch.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -q --no-deps einops timm
python - <<'PY'
import importlib.util
import subprocess
import sys
missing = [name for name in ("einops", "timm") if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
print("torch stays", __import__("torch").__version__, "cuda", __import__("torch").cuda.is_available(), flush=True)
PY

if [[ ! -d third_party/UniDepth ]]; then
  git clone --depth 1 https://github.com/lpiccinelli-eth/UniDepth third_party/UniDepth
fi
python -m pip install -q --no-deps -e third_party/UniDepth

expected=115
have=$(find data/frames -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [[ "$have" -ne "$expected" ]]; then
  rm -rf data/frames
  python scripts/fetch_clip.py --scene gaimersheim --start 12025 --stride 1 --count 115 \
    --pack-dir data/frames --cache "${CACHE:-$PWD/data/cache}"
fi

mkdir -p results
python scripts/fuse_depth.py --frames data/frames --output results --device cuda:0
echo "fuse job done. Return results/eval.json and results/summary.md"
