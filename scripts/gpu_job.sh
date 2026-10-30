#!/usr/bin/env bash
# Qwen-Drive perception + object/lane tracking on one A2D2 clip.
# GPU runner: Kaggle T4, fp16, SDPA (no flash-attn, no bf16). Internet on for HF weights.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -q -r requirements-gpu.txt
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, flush=True)"

if [[ ! -d third_party/Qwen-Drive-1.0 ]]; then
  git clone --depth 1 https://github.com/QwenLM/Qwen-Drive-1.0 third_party/Qwen-Drive-1.0
fi
# The perception CUDA kernel is bf16-only and will not run on a T4.
# On any other dtype, use the upstream PyTorch fallback instead of raising.
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
if old not in text:
    raise SystemExit("attention.py dispatch changed; refusing to patch blindly")
p.write_text(text.replace(old, new, 1))
print("patched deformable-attn fallback for fp16")
PY
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
