"""Stage 1 (GPU Bot, Sudesh t29u): WildDet3D text-prompted 3D boxes on each clip -> one npz per clip.

Uses allenai/WildDet3D build_model + preprocess (README inference API). Monocular: the checkpoint's own
LingBot-Depth is used for metric depth; no depth_gt is passed (Depth Anything V2 Small is relative depth, not
metres, so it is used only for the overlay tint in stage 2). fp16 autocast + SDPA (T4 has no bf16/FlashAttention).
"""
import argparse, json, math, os, sys, time
import cv2, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--wd3d-root", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--inputs", required=True, help="dir with <tag>.mp4 and cams.json")
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--prompts", nargs="+", default=["car", "person", "bicycle", "lane line", "lane marking", "road edge", "road curb"])
ap.add_argument("--score", type=float, default=0.3)
ap.add_argument("--score3d", type=float, default=0.1)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--budget-s", type=float, default=3.0 * 3600, help="stage-1 GPU budget; auto-stride if projected over")
ap.add_argument("--max-frames", type=int, default=0, help="smoke: cap frames per clip")
ap.add_argument("--fp16", type=int, default=1)
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.wd3d_root))
from wilddet3d import build_model, preprocess  # noqa: E402

cams = json.load(open(os.path.join(a.inputs, "cams.json")))
os.makedirs(a.outdir, exist_ok=True)
t_build = time.time()
model = build_model(checkpoint=a.ckpt, score_threshold=a.score, score_3d_threshold=a.score3d,
                    skip_pretrained=True, device=a.device)
model.eval()
print(f"model built in {time.time()-t_build:.1f}s", flush=True)
dev = torch.device(a.device)
use_cuda = dev.type == "cuda"


def count(tag):
    cap = cv2.VideoCapture(os.path.join(a.inputs, f"{tag}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    return min(n, a.max_frames) if a.max_frames else n


def npy(v):
    return v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)


KPRED = {}


def infer(rgb, K, tag=None, want_k=False):
    data = preprocess(rgb.astype(np.float32), K)
    kw = dict(images=data["images"].to(dev), intrinsics=data["intrinsics"].to(dev)[None],
              input_hw=[data["input_hw"]], original_hw=[data["original_hw"]], padding=[data["padding"]],
              input_texts=list(a.prompts))
    if want_k:
        kw["return_predicted_intrinsics"] = True
    with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=bool(a.fp16 and use_cuda)):
        res = model(**kw)
    boxes, boxes3d, scores, s2d, s3d, cids = (npy(r[0]) for r in res[:6])
    if want_k:  # log the geometry backend's own intrinsics guess, mapped back to original pixels (info only)
        try:
            kp = npy(res[7]).reshape(-1, 3, 3)[0]; ki = npy(data["intrinsics"]).reshape(3, 3)
            sc = ki[0, 0] / K[0, 0]; pl = ki[0, 2] - sc * K[0, 2]; pt = ki[1, 2] - sc * K[1, 2]
            fx = kp[0, 0] / sc; W0 = data["original_hw"][1]
            KPRED.setdefault(tag, []).append(dict(fx=round(float(fx), 1), fy=round(float(kp[1, 1] / sc), 1),
                cx=round(float((kp[0, 2] - pl) / sc), 1), cy=round(float((kp[1, 2] - pt) / sc), 1),
                hfov_deg=round(float(np.degrees(2 * np.arctan(W0 / (2 * fx)))), 1)))
        except Exception as e:  # noqa: BLE001
            KPRED.setdefault(tag, []).append(dict(err=repr(e)[:200]))
    return dict(boxes=boxes.reshape(-1, 4), boxes3d=boxes3d.reshape(-1, 10), scores=scores.reshape(-1),
                s2d=s2d.reshape(-1), s3d=s3d.reshape(-1), cids=cids.reshape(-1).astype(np.int32))


totals = {t: count(t) for t in a.tags}
N_all = sum(totals.values())
stride, t_start, done, summary = 1, time.time(), 0, {}
for ci, tag in enumerate(a.tags):
    c = cams[tag]
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], np.float64)
    cap = cv2.VideoCapture(os.path.join(a.inputs, f"{tag}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    per = {k: [] for k in ("boxes", "boxes3d", "scores", "s2d", "s3d", "cids")}
    fidx, offs, times = [], [0], []
    i = 0
    while i < totals[tag]:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            t = time.time()
            d = infer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), K, tag, want_k=len(fidx) < 5)
            if use_cuda:
                torch.cuda.synchronize(dev)
            times.append(time.time() - t); done += 1
            for k in per:
                per[k].append(d[k])
            fidx.append(i); offs.append(offs[-1] + len(d["scores"]))
            if done == 15 and ci == 0 and not a.max_frames:  # projected budget check after warm-up
                spf = float(np.median(times[3:]))
                proj = spf * N_all
                if proj > a.budget_s:
                    stride = int(math.ceil(proj / a.budget_s))
                print(f"BUDGET s/frame={spf:.2f} projected_all={proj/3600:.2f}h budget={a.budget_s/3600:.2f}h -> stride={stride}", flush=True)
            if done % 25 == 0 or done <= 3:
                mem = torch.cuda.max_memory_allocated(dev) / 2**30 if use_cuda else 0
                print(f"{tag} frame {i}/{totals[tag]} n={len(d['scores'])} {times[-1]:.2f}s peak={mem:.2f}GiB", flush=True)
        i += 1
    cap.release()
    cat = {k: (np.concatenate(v) if v else np.zeros((0,) + {"boxes": (4,), "boxes3d": (10,)}.get(k, ()), np.float32)) for k, v in per.items()}
    np.savez_compressed(os.path.join(a.outdir, f"wd3d_{tag}.npz"), frame_idx=np.array(fidx, np.int32),
                        offsets=np.array(offs, np.int64), K=K, fps=fps, stride=stride,
                        prompts=np.array(a.prompts), **cat)
    summary[tag] = dict(frames_total=totals[tag], frames_run=len(fidx), stride=stride, fps=fps,
                        boxes=int(offs[-1]), s_per_frame_median=round(float(np.median(times)), 3) if times else None,
                        wall_s=round(float(np.sum(times)), 1))
    print(tag, summary[tag], flush=True)
summary["_peak_vram_GiB"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if use_cuda else None
summary["_prompts"] = a.prompts
summary["_k_pred_first5"] = KPRED
json.dump(summary, open(os.path.join(a.outdir, "wd3d_summary.json"), "w"), indent=1)
print("STAGE1 OK", flush=True)
