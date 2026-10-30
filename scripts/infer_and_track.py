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


def load_model(vlm: str, perception: str, device: str, dtype: str, attn: str):
    import torch
    from transformers import AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    # The perception package imports the VLM. Both live in the cloned repo.
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor
    from qwen_drive_perception.modeling_perception import QwenDrivePerception

    tokenizer = AutoTokenizer.from_pretrained(vlm, trust_remote_code=True)
    head = QwenDrivePerception.from_pretrained(perception, torch_dtype=torch_dtype)
    # attach() binds the VLM. The public API takes the loaded VLM module.
    from qwen_drive import QwenDriveForPlanning

    vlm_model = QwenDriveForPlanning.from_pretrained(
        vlm,
        dtype=torch_dtype,
        attn_implementation=attn,
        trust_remote_code=True,
    )
    head = head.to(device).eval()
    vlm_model = vlm_model.to(device).eval()
    # attach() wants the inner VLM module, not the planning wrapper.
    head.attach(vlm_model.vlm, tokenizer)
    processor = PerceptionProcessor(tokenizer)
    return head, processor, PerceptionFrame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--vlm", type=str, default="Qwen/Qwen-Drive-1.0-4B")
    parser.add_argument("--perception", type=str, default="Qwen/Qwen-Drive-1.0-4B/perception")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--score", type=float, default=0.2)
    args = parser.parse_args()

    head, processor, PerceptionFrame = load_model(args.vlm, args.perception, args.device, args.dtype, args.attn)
    args.output.mkdir(parents=True, exist_ok=True)
    folders = sorted(p for p in args.frames.iterdir() if (p / "frame.json").exists())
    if not folders:
        raise SystemExit(f"no frames in {args.frames}")

    packed = []
    for folder in folders:
        frame = PerceptionFrame(folder)
        inputs, img_metas = processor(frame, device=args.device)
        result = head.infer(inputs, img_metas)
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
