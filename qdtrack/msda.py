"""Multi-scale deformable attention for non-bf16 GPUs (T4 fp16).

Upstream's CUDA kernel is bf16-only and its torch fallback mixes dtypes:
``sampling_locations`` come out fp32 (camera projection runs in fp32) while
``value`` is fp16, and ``F.grid_sample`` requires both to match. This version
samples in ``value.dtype`` (grid cast to it), accumulates the per-level
weighted sum in fp32 (like the bf16 kernel), and returns ``value.dtype``.
Levels are accumulated one at a time instead of stacked, so peak memory is one
level of samples rather than all of them. Same math as
``qwen_drive_perception.layers.multi_scale_deformable_attn_pytorch``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def msda_mixed_precision(value, value_spatial_shapes, sampling_locations, attention_weights, max_elems=1 << 27):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    dtype = value.dtype
    shapes = [(int(h), int(w)) for h, w in value_spatial_shapes.tolist()]
    value_list = value.split([h * w for h, w in shapes], dim=1)
    # Grid math in fp32, then cast for grid_sample. Points behind a camera are
    # projected with depth clamped to 1e-5 (bev_encoder.point_sampling), so
    # their fp32 coords reach ~1e5+ and become inf in fp16. inf/NaN grid values
    # are undefined behaviour in grid_sample (segfault on CPU, wild indices on
    # CUDA). With padding_mode="zeros" any |g| > 1 + 2/min(H, W) samples zero,
    # so clamping to [-2, 2] gives the same result; NaN also maps to zero.
    sampling_grids = (2 * sampling_locations.float() - 1).clamp(-2.0, 2.0).nan_to_num(nan=-2.0)
    weights = attention_weights.float().transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels, num_points
    )
    out = torch.zeros(bs * num_heads, embed_dims, num_queries, device=value.device, dtype=torch.float32)
    # Bound one chunk's fp32 temporary to ~max_elems elements (SCA has
    # bs*cams*heads x ~10^4 queries x 32 points per level).
    per_query = bs * num_heads * embed_dims * num_points
    step = max(1, min(num_queries, max_elems // max(per_query, 1)))
    for level, (h, w) in enumerate(shapes):
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, h, w)
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        for q0 in range(0, num_queries, step):
            q1 = min(q0 + step, num_queries)
            sampled = F.grid_sample(
                value_l.contiguous(), grid_l[:, q0:q1].to(dtype).contiguous(), mode="bilinear", padding_mode="zeros", align_corners=False
            )
            out[:, :, q0:q1] += (sampled.float() * weights[:, :, q0:q1, level, :]).sum(-1)
            del sampled
    out = out.view(bs, num_heads * embed_dims, num_queries).transpose(1, 2).contiguous()
    return out.to(dtype)


def install() -> None:
    """Route upstream's non-bf16 dispatch to ``msda_mixed_precision``."""
    from qwen_drive_perception import attention

    original = attention.multi_scale_deformable_attn_cuda

    def dispatch(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        if value.is_cuda and value.dtype == torch.bfloat16:
            return original(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step)
        return msda_mixed_precision(value, spatial_shapes, sampling_locations, attention_weights)

    attention.multi_scale_deformable_attn_cuda = dispatch
