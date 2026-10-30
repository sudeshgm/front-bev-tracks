#!/usr/bin/env bash
# Qwen-Drive perception + object/lane tracking on one A2D2 clip.
# GPU runner: Kaggle T4, fp16, SDPA (no flash-attn, no bf16). Internet on for HF weights.
set -euo pipefail
# Less fragmentation: voxel pooling asks for one ~3.7 GiB block.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# QD_CUDA_DEBUG=1: synchronous launches + voxel_pool logging, for a precise CUDA traceback.
if [[ "${QD_CUDA_DEBUG:-0}" == "1" ]]; then export CUDA_LAUNCH_BLOCKING=1; fi
# QD_PERCEPTION_DEVICE=cuda:0 forces the single-GPU layout (default: auto).
cd "$(dirname "$0")/.."

python -m pip install -q -r requirements-gpu.txt
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, flush=True)"

if [[ ! -d third_party/Qwen-Drive-1.0 ]]; then
  git clone --depth 1 https://github.com/QwenLM/Qwen-Drive-1.0 third_party/Qwen-Drive-1.0
fi
# The perception CUDA kernel is bf16-only and will not run on a T4.
# On any other dtype, use the upstream PyTorch fallback instead of raising.
# infer_and_track.py then routes that dispatch to qdtrack/msda.py (fp16-safe).
python - <<'PY'
from pathlib import Path
p = Path("third_party/Qwen-Drive-1.0/src/qwen_drive_perception/attention.py")
text = p.read_text()
old = '''    if value.dtype != torch.bfloat16:
        raise TypeError(
            "the ms_deform_attn kernel requires CUDA bfloat16 tensors, "
            f"got {value.dtype}; run the model in bfloat16"
        )
'''
new = '''    if value.dtype != torch.bfloat16:
        # T4 has no bf16. Same math as the off-GPU path.
        return multi_scale_deformable_attn_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )
'''
if new in text:
    print("deformable-attn fallback already patched")
    raise SystemExit(0)
if old not in text:
    raise SystemExit("attention.py dispatch changed; refusing to patch blindly")
p.write_text(text.replace(old, new, 1))
print("patched deformable-attn fallback for fp16")
PY
export PYTHONPATH="${PWD}/third_party/Qwen-Drive-1.0/src:${PYTHONPATH:-}"

# Same Gaimersheim road window, every camera frame (30 Hz): 12025..12139.
expected=115
have=$(find data/frames -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
if [[ "$have" -ne "$expected" ]]; then
  rm -rf data/frames
  python scripts/fetch_clip.py --scene gaimersheim --start 12025 --stride 1 --count 115 \
    --pack-dir data/frames --cache "${CACHE:-$PWD/data/cache}"
fi

python scripts/infer_and_track.py \
  --frames data/frames \
  --vlm "${VLM:-Qwen/Qwen-Drive-1.0-4B}" \
  --perception "${PERCEPTION:-Qwen/Qwen-Drive-1.0-4B/perception}" \
  --dtype float16 \
  --attn sdpa \
  --perception-device "${QD_PERCEPTION_DEVICE:-auto}" \
  --output results

echo "GPU job done. Return results/eval.json and results/summary.md"
