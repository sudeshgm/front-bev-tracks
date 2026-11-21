"""Official Qwen-Drive scripts/run_perception.py loop, made runnable on free T4s.

GPU Bot T4 shim (separate commit). The official script hard-codes bfloat16 and
one device; a T4 has no bf16 and one 16 GB T4 cannot hold the VLM plus voxel
pooling. This file keeps the official loop (PerceptionFrame -> PerceptionProcessor
-> QwenDrivePerception.infer -> np.savez per token) and only swaps model loading
for the patch set already proven on Kaggle 2xT4 in this repo (ex sudeshgm/qwen-drive-a2d2)
(scripts/infer_and_track.py load_model + qdtrack/msda.py): fp16, SDPA, planner
dropped, VLM on cuda:0, BEV head on cuda:1, fp16-safe deformable attention,
fp32 calibration and fp32 box decoding. No quantization.

Usage (needs PYTHONPATH with Qwen-Drive-1.0/src and this repo root;
attention.py must carry the a2d2 gpu_job.sh bf16->pytorch fallback patch):
  python run_perception_t4.py --vlm Qwen/Qwen-Drive-1.0-4B --frames rings/ --output preds/
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch


def to_np(v):
    return v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default="Qwen/Qwen-Drive-1.0-4B")
    ap.add_argument("--model", default="Qwen/Qwen-Drive-1.0-4B/perception")
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--a2d2-root", type=Path, required=True, help="root of this repo (front-bev-tracks)")
    ap.add_argument("--perception-device", default="auto")
    a = ap.parse_args()
    sys.path.insert(0, str(a.a2d2_root)); sys.path.insert(0, str(a.a2d2_root / "scripts"))
    from infer_and_track import load_model  # a2d2 T4 patch set

    t0 = time.time()
    head, processor, PerceptionFrame = load_model(a.vlm, a.model, "cuda", "float16", "sdpa", a.perception_device)
    load_s = time.time() - t0
    a.output.mkdir(parents=True, exist_ok=True)
    stats = {"load_s": round(load_s, 1), "dtype": "float16", "attn": "sdpa",
             "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "frames": {}}
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    for frame_dir in sorted(p for p in a.frames.iterdir() if p.is_dir()):
        frame = PerceptionFrame(frame_dir)
        t = time.time()
        inputs, img_metas = processor(frame, device="cuda")
        with torch.no_grad():
            result = head.infer(inputs, img_metas)
        torch.cuda.synchronize()
        dt = time.time() - t
        out = {k: to_np(v) for k, v in result.items()}
        np.savez(a.output / f"{frame.token}.npz", **out)
        sc = out.get("scores", np.zeros(0)).reshape(-1)
        stats["frames"][frame.token] = {"s": round(dt, 2), "n_boxes": int(sc.size), "n_over_0.25": int((sc >= 0.25).sum())}
        print(f"{frame.token} ({frame.dataset_type}): {sc.size} boxes, {(sc >= 0.25).sum()} >=0.25, {dt:.2f}s", flush=True)
        del inputs, result
        torch.cuda.empty_cache()
    stats["peak_mem_gib"] = {f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}
    ts = [v["s"] for v in stats["frames"].values()]
    stats["sec_per_frame_mean"] = round(float(np.mean(ts)), 2) if ts else None
    (a.output / "perception_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "frames"}), flush=True)


if __name__ == "__main__":
    main()
