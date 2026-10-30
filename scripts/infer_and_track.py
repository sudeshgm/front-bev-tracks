#!/usr/bin/env python3
"""Run Qwen-Drive perception on the packed A2D2 clip, then track objects and lanes.

T4 notes (the GPU runner): no bf16, no flash-attn. Use --dtype float16 --attn sdpa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdtrack.tracks import (  # noqa: E402
    DET_CLASS_NAMES,
    MAP_CLASS_NAMES,
    polylines_from_map,
    track_lanes,
    track_objects,
)


def _to_numpy(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _materialize(vlm: str, perception: str) -> tuple[str, str]:
    """Local directories for the VLM root and the perception head.

    A hub id cannot contain a subfolder. ``Qwen/Qwen-Drive-1.0-4B/perception``
    is rejected by huggingface_hub, which is what killed the first T4 run.
    Download the release (skipping the unused planners) and load from disk.
    A local directory pair is used as-is.
    """
    vlm_path, perc_path = Path(vlm), Path(perception)
    if vlm_path.is_dir() and perc_path.is_dir():
        return str(vlm_path), str(perc_path)
    from huggingface_hub import snapshot_download

    sub = "perception"
    prefix = vlm.rstrip("/") + "/"
    if perception.startswith(prefix):
        sub = perception[len(prefix):]
    root = Path(
        snapshot_download(
            vlm,
            ignore_patterns=["planner-sft/*", "planner-rl/*"],
        )
    )
    head = root / sub
    if not head.is_dir():
        raise SystemExit(f"perception weights not found at {head}")
    return str(root), str(head)


def _move_inplace(obj, device):
    """Move tensors to ``device``; dicts/lists are updated in place so callers
    that later read the same ``img_metas`` see consistent objects."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _move_inplace(value, device)
        return obj
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = _move_inplace(value, device)
        return obj
    if isinstance(obj, tuple):
        return tuple(_move_inplace(v, device) for v in obj)
    return obj


def _pick_perception_device(device: str, requested: str) -> str:
    """The 4B VLM fills most of a 16 GB T4 and voxel pooling then OOMs on the
    same card. With two GPUs (Kaggle T4 x2) run the BEV head on the second."""
    if requested != "auto":
        return requested
    import torch

    if device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return "cuda:1"
    return device


def load_model(vlm: str, perception: str, device: str, dtype: str, attn: str, perception_device: str = "auto"):
    import torch
    from transformers import AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    # The perception package imports the VLM. Both live in the cloned repo.
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor
    from qwen_drive_perception.modeling_perception import QwenDrivePerception
    from qwen_drive import QwenDriveForPlanning

    vlm_dir, head_dir = _materialize(vlm, perception)
    tokenizer = AutoTokenizer.from_pretrained(vlm_dir, trust_remote_code=True)
    # dtype= (not torch_dtype=). Drop the planning expert: it is unused and
    # does not fit beside the 4B VLM on a 16 GB T4.
    holder = QwenDriveForPlanning.from_pretrained(
        vlm_dir,
        dtype=torch_dtype,
        attn_implementation=attn,
        trust_remote_code=True,
    )
    vlm_mod = holder.vlm
    if hasattr(holder, "planning_expert"):
        del holder.planning_expert
    del holder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    head = QwenDrivePerception.from_pretrained(head_dir, dtype=torch_dtype)
    perc_device = _pick_perception_device(device, perception_device)
    head = head.to(perc_device).eval()
    vlm_mod = vlm_mod.to(device).eval()
    if perc_device != device:
        # infer() runs the VLM, then calls bev_modeling with features and
        # img_metas tensors still on the VLM device. Move them across.
        def _to_perc(module, args, kwargs):
            return _move_inplace(tuple(args), perc_device), _move_inplace(dict(kwargs), perc_device)

        head.bev_modeling.register_forward_pre_hook(_to_perc, with_kwargs=True)
    print(f"vlm on {device}, perception head on {perc_device}", flush=True)
    # attach() takes the inner VLM and the PerceptionProcessor, not a tokenizer.
    processor = PerceptionProcessor(tokenizer)
    head.attach(vlm_mod, processor)
    return head, processor, PerceptionFrame


def _empty_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--vlm", type=str, default="Qwen/Qwen-Drive-1.0-4B")
    parser.add_argument("--perception", type=str, default="Qwen/Qwen-Drive-1.0-4B/perception")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--perception-device", default="auto", help="auto: cuda:1 when two GPUs exist")
    parser.add_argument("--score", type=float, default=0.2)
    args = parser.parse_args()

    head, processor, PerceptionFrame = load_model(
        args.vlm, args.perception, args.device, args.dtype, args.attn, args.perception_device
    )
    args.output.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in args.frames.iterdir() if (p / "frame.json").exists())
    if not folders:
        raise SystemExit(f"no frames in {args.frames}")

    packed = []
    for folder in folders:
        frame = PerceptionFrame(folder)
        inputs, img_metas = processor(frame, device=args.device)
        result = head.infer(inputs, img_metas)
        del inputs
        boxes = _to_numpy(result["boxes"]).reshape(-1, 9)
        scores = _to_numpy(result["scores"]).reshape(-1)
        labels = _to_numpy(result["labels"]).reshape(-1).astype(np.int64)
        occ = _to_numpy(result["occ"]).astype(np.uint8)
        seg = _to_numpy(result["map"]).astype(np.uint8)
        np.savez(args.output / f"{frame.token}.npz", boxes=boxes, scores=scores, labels=labels, occ=occ, map=seg)
        meta = json.loads((folder / "frame.json").read_text())
        maps = {}
        for name in ("road_line", "road_edge"):
            maps[name] = [line.tolist() for line in polylines_from_map(seg, name)]
        packed.append(
            {
                "token": frame.token,
                "t": meta["pose"]["t"],
                "pose": meta["pose"],
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "maps": maps,
            }
        )
        kept = int((scores >= args.score).sum()) if len(scores) else 0
        del result
        _empty_cache()
        print(f"{frame.token}: {len(scores)} boxes, {kept} over {args.score}, map {seg.shape}", flush=True)

    objects = track_objects(packed, score_min=args.score)
    lanes = track_lanes(packed)
    eval_doc = {
        "model": args.vlm,
        "perception": args.perception,
        "dtype": args.dtype,
        "attn": args.attn,
        "score_min": args.score,
        "classes": list(DET_CLASS_NAMES),
        "map_classes": list(MAP_CLASS_NAMES),
        "object_tracks": objects,
        "lane_tracks": lanes,
        "frames": [
            {
                "token": frame["token"],
                "detections": [
                    {
                        "label": int(label),
                        "name": DET_CLASS_NAMES[int(label)] if 0 <= int(label) < len(DET_CLASS_NAMES) else "?",
                        "score": round(float(score), 4),
                        "box": [round(float(v), 4) for v in box],
                    }
                    for box, score, label in zip(frame["boxes"], frame["scores"], frame["labels"])
                    if float(score) >= args.score
                ],
            }
            for frame in packed
        ],
    }
    dest = args.output / "eval.json"
    dest.write_text(json.dumps(eval_doc))
    summary = args.output / "summary.md"
    summary.write_text(
        "\n".join(
            [
                "# Qwen-Drive on A2D2",
                "",
                f"- frames: {len(packed)}",
                f"- object tracks: {len(objects)}",
                f"- lane tracks: {len(lanes)} (road_line + road_edge)",
                f"- score floor: {args.score}",
                "",
                "Load `results/eval.json` in the evaluation UI.",
                "",
            ]
        )
    )
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
