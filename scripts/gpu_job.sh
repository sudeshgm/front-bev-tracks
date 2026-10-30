#!/usr/bin/env bash
# Qwen-Drive perception + object/lane tracking on one A2D2 clip.
# GPU runner: Kaggle T4, fp16, SDPA (no flash-attn, no bf16). Internet on for HF weights.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -q -r requirements-gpu.txt

if [[ ! -d third_party/Qwen-Drive-1.0 ]]; then
  git clone --depth 1 https://github.com/QwenLM/Qwen-Drive-1.0 third_party/Qwen-Drive-1.0
fi
export PYTHONPATH="${PWD}/third_party/Qwen-Drive-1.0/src:${PYTHONPATH:-}"

# Pack the clip (downloads the Audi frames). Skip if a previous attempt finished.
if [[ ! -d data/frames/f000008472/images ]]; then
  python scripts/fetch_clip.py --pack-dir data/frames --cache "${CACHE:-$PWD/data/cache}"
fi

python scripts/infer_and_track.py \
  --frames data/frames \
  --vlm "${VLM:-Qwen/Qwen-Drive-1.0-4B}" \
  --perception "${PERCEPTION:-Qwen/Qwen-Drive-1.0-4B/perception}" \
  --dtype float16 \
  --attn sdpa \
  --output results

echo "GPU job done. Return results/eval.json and results/summary.md"
