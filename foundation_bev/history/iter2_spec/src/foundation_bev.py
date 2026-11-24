#!/usr/bin/env python3
"""Foundation-model BEV, iter2: WildDet3D + SAM 2.1 hiera-small + Depth Anything V2 Small.

No TwinLiteNet, no YOLO, no Grounding DINO. (WildDet3D imports the sam3 *code* as its own internal
dependency; SAM 3 is not run as a separate segmenter.)
Output: side-by-side front + BEV. BEV shows only class-coloured trajectories (car/person/bicycle),
lane-line curves (green), road-edge curves (orange, separate from lanes) and the ego path.
No depth maps, no edge maps, no masks are drawn.

iter2 changes vs iter1 (see SPEC.md):
- stage-1 detections are required (--require-dets): exit 7 if the npz is missing or has 0 object boxes
- stride frames: Kalman predict only (iter1 reused stale detections)
- per-clip ground model (pitch from median vanishing row + DAV2 plane fit, median DAV2 scale) for
  every ground lift; no per-frame depth scaling of lines
- ByteTrack-style two-stage association + CV Kalman (per-class noise) + RTS + re-link <=1.5 m,
  min length 5, coast 10, majority-vote class, stationary hold
- lanes: WD3D lane boxes -> SAM masks + top-hat thin stripes -> skeleton -> BEV -> lateral clusters
  -> RANSAC x(z) poly2 -> EMA(0.3) with 0.5 m gate, hold 15 frames, centred 7-frame window
- edges: SAM drivable area, lower envelope from the ego corridor, object boxes masked, rail pairs
  rejected, burned-in overlay masked, RANSAC poly2, same line tracker (one per side)
- the front view shows the fitted curves projected back (front and BEV agree); iter1's
  smooth_poly zero-padding bug (hooks toward the image origin) is gone
- two passes: pass 1 = models + raw observations, pass 2 = tracking/fitting, pass 3 = render (stream)
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

OBJ_PROMPTS = ("car", "person", "bicycle")
LANE_PROMPTS = ("lane line", "lane marking")
EDGE_PROMPTS = ("road edge", "road curb")

# BGR
COL_CLASS = {
    "car": (0, 180, 255),
    "person": (200, 80, 255),
    "bicycle": (255, 200, 0),
}
COL_LANE = (80, 210, 90)
COL_EDGE = (40, 140, 255)
COL_EGO = (230, 230, 230)
COL_PATH = (200, 200, 200)

MAX_DIST = 40.0
BEV_W = 400


def load_depth_anything(device, weights_dir: Path):
    from depth_anything_v2.dpt import DepthAnythingV2
    import torch

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    ckpt = Path(weights_dir) / "depth_anything_v2_vits.pth"
    if not ckpt.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            "depth-anything/Depth-Anything-V2-Small",
            "depth_anything_v2_vits.pth",
            local_dir=str(weights_dir),
        )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


def load_sam_predictor(device, weights_dir: Path, config: str, ckpt_name: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt = Path(weights_dir) / ckpt_name
    sam = build_sam2(config, str(ckpt), device=device)
    print("SAM loaded:", config, ckpt_name, "on", device, flush=True)
    return SAM2ImagePredictor(sam)


def disparity_to_metric(disp, fy, cy, cam_height, h, w, eps=1e-3, y0=None, y1=None):
    """Fit z = s/disp from a flat-ground patch in the road band."""
    if y0 is None:
        v0, v1 = int(0.80 * h), int(0.97 * h)
    else:
        v0, v1 = int(y0 + 0.55 * (y1 - y0)), int(y0 + 0.95 * (y1 - y0))
        v0, v1 = max(0, v0), min(h, max(v0 + 8, v1))
    u0, u1 = int(0.35 * w), int(0.65 * w)
    rows = np.arange(v0, v1, dtype=np.float32)[:, None]
    zg = fy * cam_height / np.maximum(rows - cy, 1.0)
    patch = disp[v0:v1, u0:u1]
    ok = (patch > eps) & ((rows - cy) > 1.0)
    if int(ok.sum()) < 50:
        return None, None
    s = float(np.median((zg * patch)[ok]))
    return s / np.maximum(disp, eps), s


def flat_ground_depth(h, w, fy, cy, cam_height):
    depth = np.zeros((h, w), np.float32)
    vs = np.arange(h, dtype=np.float32)[:, None]
    z = fy * cam_height / np.maximum(vs - cy, 1.0)
    z = np.clip(z, 0.5, MAX_DIST * 1.5)
    depth[:] = z
    depth[: max(0, int(math.floor(cy + 1)))] = 0
    return depth


def lift_uv(us, vs, depth, fx, fy, cx, cy):
    us = np.asarray(us, np.float32)
    vs = np.asarray(vs, np.float32)
    if us.size == 0:
        return np.zeros((0, 2), np.float32)
    h, w = depth.shape
    ui = np.clip(np.round(us).astype(int), 0, w - 1)
    vi = np.clip(np.round(vs).astype(int), 0, h - 1)
    z = depth[vi, ui]
    ok = (z > 0.5) & (z < MAX_DIST)
    X = (us - cx) * z / fx
    Z = z
    out = np.stack([X, Z], axis=1)
    out[~ok] = np.nan
    return out


def sane_wd3d_xz(b3, cam_height):
    """WildDet3D boxes3d: centre xyz (OpenCV), w,l,h, quat wxyz. Return (x_right, z_fwd, x,y,z) or None."""
    if b3 is None or len(b3) < 6:
        return None
    x, y, z, w, l, h = [float(v) for v in b3[:6]]
    if not (0.5 < z < 80.0) or abs(x) > 40.0 or not math.isfinite(x + y + z):
        return None
    if h > 0:
        y_bottom = y + 0.5 * h
        if abs(y_bottom - cam_height) > 3.0 and abs(y_bottom) > 4.0:
            # still accept x,z if range is sane
            pass
    return (x, z, x, y, z)


def trapezoid_road(h, w, y0, y1):
    mask = np.zeros((h, w), np.uint8)
    top_w = int(0.35 * w)
    bot_w = int(0.95 * w)
    y0, y1 = int(y0), int(y1)
    pts = np.array(
        [
            [(w - top_w) // 2, y0],
            [(w + top_w) // 2, y0],
            [(w + bot_w) // 2, y1 - 1],
            [(w - bot_w) // 2, y1 - 1],
        ],
        np.int32,
    )
    cv2.fillConvexPoly(mask, pts, 1)
    return mask


def useful_prompt_box(box, h, w, y0, y1):
    x1, y1b, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1b)
    area = bw * bh
    cx, cy = (x1 + x2) / 2.0, (y1b + y2) / 2.0
    if area < 0.0004 * h * w or area > 0.12 * h * w:
        return False
    if not (y0 <= cy <= y1):
        return False
    if min(bw, bh) > 0.35 * min(h, w):
        return False
    return True


def sam_masks_from_boxes(predictor, boxes, h, w, y0, y1):
    masks = []
    if predictor is None or not boxes:
        return masks
    for box in boxes:
        try:
            m, sc, _ = predictor.predict(
                box=np.asarray(box, np.float32),
                multimask_output=False,
            )
        except Exception:
            continue
        if m is None or len(m) == 0:
            continue
        seg = m[0].astype(np.uint8)
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        area = int(seg.sum())
        if area < 30 or area > 0.15 * h * w:
            continue
        if seg[int(y0) : int(y1)].sum() < 0.5 * area:
            continue
        masks.append(seg)
    return masks


def sam_road_mask(predictor, h, w, y0, y1):
    if predictor is None:
        return trapezoid_road(h, w, y0, y1)
    ys = np.linspace(y0 + 0.15 * (y1 - y0), y1 - 0.08 * (y1 - y0), 3)
    xs = [0.50 * w, 0.38 * w, 0.62 * w]
    pts, labs = [], []
    for i, y in enumerate(ys):
        pts.append([xs[0] if i == 0 else xs[i % 3], y])
        labs.append(1)
    # negative points in the sky / far above the road band
    pts.append([0.5 * w, max(8.0, 0.15 * h)])
    labs.append(0)
    try:
        m, sc, _ = predictor.predict(
            point_coords=np.asarray(pts, np.float32),
            point_labels=np.asarray(labs, np.int32),
            multimask_output=True,
        )
    except Exception:
        return trapezoid_road(h, w, y0, y1)
    best, best_s = None, -1
    for seg, s in zip(m, sc):
        seg = seg.astype(np.uint8)
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        area = int(seg.sum())
        bot = int(seg[int(y0) : int(y1)].sum())
        top = int(seg[: int(0.25 * h)].sum())
        if area < 0.04 * h * w or area > 0.75 * h * w:
            continue
        if bot < 0.5 * area:
            continue
        if top > 0.25 * area:
            continue
        score = float(s) + 0.2 * (bot / max(1, area))
        if score > best_s:
            best, best_s = seg, score
    return best if best is not None else trapezoid_road(h, w, y0, y1)


def union_masks(masks, h, w):
    out = np.zeros((h, w), np.uint8)
    for m in masks:
        out = np.maximum(out, (m > 0).astype(np.uint8))
    return out


def bev_scale(h, bev_w=BEV_W, max_dist=MAX_DIST, lat=15.0):
    return float(min((h - 30) / max_dist, (bev_w / 2 - 10) / lat))


def to_bev_px(xz, h, bev_w, scale):
    x, z = float(xz[0]), float(xz[1])
    u = int(round(bev_w / 2 + x * scale))
    v = int(round(h - 20 - z * scale))
    return u, v


def load_dets(path):
    if not path:
        return None
    if not Path(path).exists():
        return None
    D = np.load(path, allow_pickle=False)
    prompts = [str(p) for p in D["prompts"]]
    by = {}
    offs = D["offsets"]
    for k, fi in enumerate(D["frame_idx"]):
        sl = slice(int(offs[k]), int(offs[k + 1]))
        by[int(fi)] = {
            "boxes": D["boxes"][sl],
            "boxes3d": D["boxes3d"][sl],
            "scores": D["scores"][sl],
            "cids": D["cids"][sl],
        }
    return {"prompts": prompts, "by": by, "K": D["K"], "stride": int(D["stride"]) if "stride" in D.files else 1}


def write_video(path, frames, fps, size):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".mp4v.mp4")
    wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not wr.isOpened():
        raise RuntimeError(f"VideoWriter failed for {tmp}")
    for im in frames:
        wr.write(im)
    wr.release()
    # prefer H.264 when ffmpeg is present
    import shutil
    import subprocess

    if shutil.which("ffmpeg"):
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "20", "-movflags", "+faststart", "-an", str(path),
            ],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and path.exists() and path.stat().st_size > 0:
            tmp.unlink(missing_ok=True)
            return str(path)
        print("ffmpeg h264 failed, keeping mp4v:", r.stderr[-400:] if r.stderr else r.returncode, flush=True)
    tmp.replace(path)
    return str(path)


def count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def pick_still_frames(n, busy):
    """2-3 frames: early, mid, and the busiest (curve/traffic) if distinct."""
    if n <= 0:
        return []
    mid = n // 2
    early = max(0, n // 5)
    busy_i = int(np.argmax(busy)) if busy else mid
    picks = []
    for i in (early, mid, busy_i):
        i = int(np.clip(i, 0, n - 1))
        if i not in picks:
            picks.append(i)
    if len(picks) < 2 and n > 1:
        picks.append(n - 1)
    return sorted(picks)[:3]


# ============================================================================ iter2 additions
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bev_smooth as BS  # noqa: E402

OBJ_LEN = {"car": 4.2, "person": 0.5, "bicycle": 1.7}


def sam_road_mask2(predictor, h, w, y0, y1, horizon_v, neg_boxes=()):
    """Positive points on the ego corridor (centre column, lower band), negatives in the sky and on
    detected objects, so parked cars / tram tracks are less often swallowed into the road."""
    if predictor is None:
        return trapezoid_road(h, w, max(y0, horizon_v + 6), y1)
    lo = max(y0, horizon_v + 0.25 * (y1 - horizon_v))
    ys = np.linspace(lo, y1 - 0.06 * (y1 - y0), 4)
    # two columns either side of centre: a burned-in centre line (A2D2/PhysicalAI HUD) at x=w/2 otherwise
    # gets segmented instead of the road; positives on the ego corridor only
    pts = [[xf * w, y] for y in ys for xf in (0.44, 0.56)]
    labs = [1] * len(pts)
    pts.append([0.5 * w, max(8.0, 0.5 * max(horizon_v, 20.0))])
    labs.append(0)
    for b in list(neg_boxes)[:6]:
        pts.append([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])])
        labs.append(0)
    try:
        m, sc, _ = predictor.predict(point_coords=np.asarray(pts, np.float32),
                                     point_labels=np.asarray(labs, np.int32), multimask_output=True)
    except Exception:  # noqa: BLE001
        return trapezoid_road(h, w, max(y0, horizon_v + 6), y1)
    best, best_s = None, -1e9
    for seg, s in zip(m, sc):
        seg = seg.astype(np.uint8)
        if seg.shape[:2] != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST)
        area = int(seg.sum())
        above = int(seg[: max(0, int(horizon_v - 4))].sum())
        if area < 0.02 * h * w or area > 0.7 * h * w or above > 0.05 * area:
            continue
        score = float(s) - 2.0 * above / max(area, 1)
        if score > best_s:
            best, best_s = seg, score
    if best is None:
        return trapezoid_road(h, w, max(y0, horizon_v + 6), y1)
    best[: max(0, int(horizon_v))] = 0
    return best


def exact_dets(pack, frame):
    """iter1 reused the previous frame's boxes on stride frames; iter2 returns None (predict only)."""
    if pack is None:
        return None
    return pack["by"].get(frame)


def frame_objects(pack, frame, h, w, score_min=0.25):
    raw = exact_dets(pack, frame)
    if raw is None:
        return None, [], []
    prompts = pack["prompts"]
    objs, lane_boxes, edge_boxes = [], [], []
    for i, cid in enumerate(raw["cids"]):
        cid = int(cid)
        name = prompts[cid] if 0 <= cid < len(prompts) else ""
        sc = float(raw["scores"][i])
        box = [float(v) for v in raw["boxes"][i]]
        if name in LANE_PROMPTS:
            lane_boxes.append(box)
            continue
        if name in EDGE_PROMPTS:
            edge_boxes.append(box)
            continue
        if name not in OBJ_PROMPTS or sc < score_min:
            continue
        b3 = raw["boxes3d"][i]
        objs.append({"cls": name, "score": sc, "box": box,
                     "b3": [float(v) for v in b3[:7]] if b3 is not None and len(b3) >= 6 else None})
    return objs, lane_boxes, edge_boxes


def obj_bev(o, gm: "BS.GroundModel", h, mode="fuse"):
    """BEV centre of one object. WD3D 3D centre, ground contact (+half length), or a fusion."""
    x1, y1, x2, y2 = o["box"]
    vb = min(y2, h - 2.0)
    gc = gm.lift([0.5 * (x1 + x2)], [vb])[0]
    if np.isfinite(gc).all():
        gc = (float(gc[0]), float(gc[1]) + 0.5 * OBJ_LEN.get(o["cls"], 1.0))
    else:
        gc = None
    wd = None
    if o.get("b3") is not None:
        got = sane_wd3d_xz(o["b3"], gm.h)
        if got is not None:
            wd = (got[0], got[1])
    truncated = y2 >= h - 3
    if mode == "wd3d" or (mode == "fuse" and (gc is None or truncated)):
        return (wd, "wd3d") if wd is not None else ((gc, "ground") if gc is not None else (None, None))
    if mode == "ground" or wd is None:
        return (gc, "ground") if gc is not None else ((wd, "wd3d") if wd is not None else (None, None))
    # fuse: inverse variance when both agree within 35 % in range, else trust WD3D
    if abs(wd[1] - gc[1]) > 0.35 * max(wd[1], 1.0):
        return wd, "wd3d"
    la, lo_g = BS.meas_sigma(gc[1], "ground")
    lb, lo_w = BS.meas_sigma(wd[1], "wd3d")
    wg, ww = 1 / lo_g ** 2, 1 / lo_w ** 2
    z = (wg * gc[1] + ww * wd[1]) / (wg + ww)
    x = (gc[0] / la ** 2 + wd[0] / lb ** 2) / (1 / la ** 2 + 1 / lb ** 2)
    return (float(x), float(z)), "fuse"


def row_centres(mask, top, bottom, min_run=1, max_run=40, step=2):
    """Medial points of thin runs per row (fast skeleton for near-vertical stripes)."""
    pts = []
    for v in range(int(bottom) - 1, int(top), -step):
        cols = np.flatnonzero(mask[v])
        if cols.size == 0:
            continue
        splits = np.where(np.diff(cols) > 2)[0] + 1
        for run in np.split(cols, splits):
            if min_run <= len(run) <= max_run:
                pts.append((float(run.mean()), float(v)))
    return np.asarray(pts, np.float64).reshape(-1, 2)


def subsample(a, n):
    if len(a) <= n:
        return a
    idx = np.linspace(0, len(a) - 1, n).astype(int)
    return a[idx]


def observe_frame(bgr, rgb, predictor, objs, lane_boxes, edge_boxes, h, w, y0, y1, gm0, overlay, u_ego, use_rails=False):
    """Pass-1 raw observations in pixels (lifted in pass 2 with the per-clip ground)."""
    hv = gm0.horizon_v
    if predictor is not None:
        predictor.set_image(rgb)
    obj_boxes = [o["box"] for o in objs]
    road = sam_road_mask2(predictor, h, w, y0, y1, hv, neg_boxes=[b for b in obj_boxes if b[3] > hv])
    road = BS.fill_road(road, w)
    k_l = max(5, (w // 60) | 1)
    road_lanes = cv2.dilate(road, cv2.getStructuringElement(cv2.MORPH_RECT, (k_l, 1)))  # include boundary paint
    occl = BS.box_mask(obj_boxes, h, w)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if use_rails:
        rails, n_rail = BS.rail_mask(gray, gm0, y0, y1, h, w)
    else:
        rails, n_rail = np.zeros((h, w), np.uint8), 0
    bad = (occl > 0) | (rails > 0)
    if overlay is not None:
        bad |= overlay > 0
    # lanes: SAM masks from WD3D lane boxes (skeleton) + thin bright stripes (row medial points)
    lane_px = []
    useful_lb = [b for b in lane_boxes if useful_prompt_box(b, h, w, y0, y1)]
    if predictor is not None and useful_lb:
        lm = union_masks(sam_masks_from_boxes(predictor, useful_lb, h, w, y0, y1), h, w)
        lm[bad] = 0
        lm[: int(hv + 6)] = 0
        sk = BS.thin(lm)
        vv, uu = np.nonzero(sk)
        lane_px.append(np.stack([uu, vv], 1).astype(np.float64))
    tb = BS.thin_bright_mask(bgr, road_lanes, y0, y1, hv)
    tb[bad] = 0
    lane_px.append(row_centres(tb, max(y0, hv + 6), y1))
    lane_px = np.concatenate([p for p in lane_px if len(p)], 0) if any(len(p) for p in lane_px) else np.zeros((0, 2))
    # edges: lower envelope of the drivable area around the ego corridor
    rails_overlay = rails.copy()
    if overlay is not None:
        rails_overlay = np.maximum(rails_overlay, overlay)
    occl_scan = occl if overlay is None else np.maximum(occl, overlay)  # HUD pixels: filled, never an edge
    left, right = BS.edge_points_lower_envelope(road, occl_scan, rails, y0, y1, hv, u_ego)
    useful_eb = [b for b in edge_boxes if useful_prompt_box(b, h, w, y0, y1)]
    if predictor is not None and useful_eb:
        em = union_masks(sam_masks_from_boxes(predictor, useful_eb, h, w, y0, y1), h, w)
        em[bad] = 0
        em[: int(hv + 6)] = 0
        vv, uu = np.nonzero(BS.thin(em))
        if len(uu):
            pts = np.stack([uu, vv], 1).astype(np.float64)
            lpts, rpts = pts[pts[:, 0] < u_ego], pts[pts[:, 0] >= u_ego]
            left = np.concatenate([left.reshape(-1, 2), lpts]) if len(lpts) else left
            right = np.concatenate([right.reshape(-1, 2), rpts]) if len(rpts) else right
    vp = BS.vp_row_from_lines([left, right], h) if len(left) and len(right) else []
    return {
        "lane_px": subsample(lane_px, 1500).astype(np.float32),
        "edge_px": [subsample(np.asarray(left).reshape(-1, 2), 400).astype(np.float32),
                    subsample(np.asarray(right).reshape(-1, 2), 400).astype(np.float32)],
        "vp": vp, "n_rail_pairs": n_rail, "road": road,
    }


def fit_frame_lines(obs, gm, lane_prior, args):
    lane_bev = gm.lift(obs["lane_px"][:, 0], obs["lane_px"][:, 1]) if len(obs["lane_px"]) else np.zeros((0, 2))
    lane_bev = lane_bev[np.isfinite(lane_bev).all(1)] if len(lane_bev) else lane_bev
    lane_bev = lane_bev[(np.abs(lane_bev[:, 0]) < 12) & (lane_bev[:, 1] < args.max_line_z)] if len(lane_bev) else lane_bev
    fits = []
    for cl in BS.cluster_lateral(lane_bev, lane_prior, gap=0.7, min_pts=args.min_lane_pts):
        f = BS.ransac_poly2(cl, thr=0.2, min_inl=args.min_lane_pts, min_span=5.0)
        if f is not None:
            fits.append(f)
    edge_fits = []
    for side, px in zip((-1, 1), obs["edge_px"]):
        if len(px) < 8:
            continue
        b = gm.lift(px[:, 0], px[:, 1])
        b = b[np.isfinite(b).all(1)]
        b = b[(b[:, 0] * side > 0.5) & (b[:, 1] < args.max_line_z)]
        f = BS.ransac_poly2(b, thr=0.35, min_inl=10, min_span=6.0)
        if f is not None and BS.x_at(f["c"], max(f["zmin"], 6.0)) * side > 0.5:  # side test at the near end, not z=0
            f["side"] = side
            edge_fits.append(f)
    return fits, edge_fits, lane_bev


class EdgePair:
    """One LineTracker per side so the left and right edges can never swap."""

    def __init__(self, args):
        self.t = {s: BS.LineTracker("edge", alpha=args.alpha, gate=args.gate, hold=args.hold, max_tracks=1,
                                    motion=args.line_motion)
                  for s in (-1, 1)}

    def step(self, frame, fits, ds):
        live = []
        for s, tr in self.t.items():
            live += [dict(r, side=s) for r in tr.step(frame, [f for f in fits if f.get("side") == s], ds)]
        return live

    @property
    def hist(self):
        out = {}
        for s, tr in self.t.items():
            for k, v in tr.hist.items():
                out[(s, k)] = v
        return out


def draw_bev2(h, now, tracks_by_frame, lanes, edges, speed, fps, tail, bev_w=BEV_W):
    panel = np.full((h, bev_w, 3), 22, np.uint8)
    scale = bev_scale(h, bev_w)
    for metre in (10, 20, 30, 40):
        y = int(round(h - 20 - metre * scale))
        if 0 <= y < h:
            cv2.line(panel, (0, y), (10, y), (80, 80, 80), 1)
            cv2.putText(panel, f"{metre}m", (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1, cv2.LINE_AA)

    def curve(rec, col):
        pts = BS.sample_curve(rec, 0.5, MAX_DIST)
        if len(pts) < 2:
            return
        uv = [to_bev_px(p, h, bev_w, scale) for p in pts]
        cv2.polylines(panel, [np.asarray(uv, np.int32)], False, col, 2, cv2.LINE_AA)

    for r in edges:
        curve(r, COL_EDGE)
    for r in lanes:
        curve(r, COL_LANE)
    travel_now = speed * now / fps
    for tr in tracks_by_frame:
        rows = [r for r in tr["rows"] if now - tail <= r["frame"] <= now]
        if not rows:
            continue
        col = COL_CLASS.get(tr["cls"], (200, 200, 200))
        pts = [to_bev_px((r["smooth"][0], r["smooth"][1] - travel_now), h, bev_w, scale) for r in rows]
        if len(pts) >= 2:
            cv2.polylines(panel, [np.asarray(pts, np.int32)], False, col, 2, cv2.LINE_AA)
        if rows[-1]["frame"] == now:
            u, v = pts[-1]
            if 0 <= u < bev_w and 0 <= v < h:
                cv2.circle(panel, (u, v), 5, col, -1, cv2.LINE_AA)
                cv2.circle(panel, (u, v), 5, (0, 0, 0), 1, cv2.LINE_AA)
    # ego path: travelled path behind the ego (straight, zero yaw) + ego marker
    back = [to_bev_px((0.0, -k * 0.5), h, bev_w, scale) for k in range(0, 40)]
    back = [p for p in back if 0 <= p[1] < h]
    if len(back) >= 2:
        cv2.polylines(panel, [np.asarray(back, np.int32)], False, COL_PATH, 2, cv2.LINE_AA)
    nose = to_bev_px((0.0, 2.0), h, bev_w, scale)
    left = to_bev_px((-0.9, -0.5), h, bev_w, scale)
    right = to_bev_px((0.9, -0.5), h, bev_w, scale)
    cv2.fillConvexPoly(panel, np.array([nose, left, right], np.int32), COL_EGO)
    return panel


def draw_front2(img, gm, now, tracks, lanes, edges, front_boxes=True):
    out = img.copy()
    h, w = out.shape[:2]

    def curve(rec, col):
        pts = BS.sample_curve(rec, 0.5, MAX_DIST)
        if len(pts) < 2:
            return
        uv = gm.project(pts)
        uv = uv[np.isfinite(uv).all(1)]
        uv = uv[(uv[:, 1] > gm.horizon_v) & (uv[:, 0] > -50) & (uv[:, 0] < w + 50)]
        if len(uv) >= 2:
            cv2.polylines(out, [uv.astype(np.int32).reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)

    for r in edges:
        curve(r, COL_EDGE)
    for r in lanes:
        curve(r, COL_LANE)
    if front_boxes:
        for tr in tracks:
            row = next((r for r in tr["rows"] if r["frame"] == now and not r["coast"] and r["box"] is not None), None)
            if row is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in row["box"]]
            col = COL_CLASS.get(tr["cls"], (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(out, f"{tr['cls']}#{tr['id']}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


class StreamWriter:
    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_suffix(".mp4v.mp4")
        self.wr = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.wr.isOpened():
            raise RuntimeError(f"VideoWriter failed for {self.tmp}")

    def write(self, im):
        self.wr.write(im)

    def close(self):
        import shutil
        import subprocess

        self.wr.release()
        if shutil.which("ffmpeg"):
            r = subprocess.run(["ffmpeg", "-y", "-i", str(self.tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-crf", "20", "-movflags", "+faststart", "-an", str(self.path)], capture_output=True, text=True)
            if r.returncode == 0 and self.path.exists() and self.path.stat().st_size > 0:
                self.tmp.unlink(missing_ok=True)
                return str(self.path)
        self.tmp.replace(self.path)
        return str(self.path)


def lane_width_check(lane_hist_frames, edge_live_frames):
    """Median spacing of adjacent lane curves and edge-to-edge width at z=10 m.
    NOTE: on a flat ground with square pixels, lateral metres = (u-cx)*H/(v-v_h): independent of fx.
    A lane-width mismatch therefore points at camera height / horizon (pitch), not at fx."""
    sp, ew = [], []
    for lanes in lane_hist_frames:
        xs = sorted(BS.x_at(r["c"], 10.0) for r in lanes if r["zmin"] <= 12 and r["zmax"] >= 8)
        sp += [b - a for a, b in zip(xs, xs[1:]) if 2.3 < b - a < 5.0]
    for edges in edge_live_frames:
        L = [BS.x_at(r["c"], 10.0) for r in edges if r.get("side") == -1]
        R = [BS.x_at(r["c"], 10.0) for r in edges if r.get("side") == 1]
        if L and R:
            ew.append(R[0] - L[0])
    return {"lane_spacing_z10_median_m": float(np.median(sp)) if sp else None, "n_lane_pairs": len(sp),
            "edge_width_z10_median_m": float(np.median(ew)) if ew else None,
            "suggested_height_scale_for_3.5m": (3.5 / float(np.median(sp))) if sp else None}


def run_clip2(args):
    cam = json.loads(Path(args.cam_json).read_text()) if args.cam_json else {}
    if args.tag and args.tag in cam:
        cam = cam[args.tag]
    elif "fx" not in cam and len(cam) == 1:
        cam = next(iter(cam.values()))
    if args.hfov_deg:
        cap0 = cv2.VideoCapture(str(args.video)); W0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)); cap0.release()
        f = 0.5 * W0 / math.tan(math.radians(args.hfov_deg) / 2)
        cam = dict(cam, fx=f, fy=f * float(cam["fy"]) / float(cam["fx"]))
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    cam_h = float(args.cam_height or cam.get("height", 1.5))
    speed = float(args.speed)
    road_y0f = float(cam.get("road_y0", 0.4))
    road_y1f = float(cam.get("road_y1", 0.98))
    tag = args.tag or Path(args.video).stem

    pack = load_dets(args.dets)
    n_obj_dets = 0
    if pack is not None:
        objp = {i for i, p in enumerate(pack["prompts"]) if p in OBJ_PROMPTS}
        n_obj_dets = int(sum(int(np.isin(v["cids"], list(objp)).sum()) for v in pack["by"].values()))
    if args.require_dets and (pack is None or n_obj_dets == 0):
        print(f"ERROR: --require-dets but dets missing or 0 object boxes ({args.dets!r}, n={n_obj_dets})", flush=True)
        sys.exit(7)

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    y0, y1 = road_y0f * h, road_y1f * h
    ds = speed / fps
    p0 = math.atan((cy - float(cam["horizon_v"])) / fy) if cam.get("horizon_v") is not None else 0.0
    gm0 = BS.GroundModel(fx, fy, cx, cy, cam_h, p0)
    use_rails = bool(cam.get("rails", False)) if args.rails < 0 else bool(args.rails)

    depth_model = predictor = None
    if not args.smoke:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available, abort (not a smoke run)", flush=True)
            sys.exit(2)
        depth_model = load_depth_anything(args.device, Path(args.weights))
        predictor = load_sam_predictor(args.device, Path(args.weights), args.sam_config, args.sam_ckpt)

    # pass 0: burned-in overlay mask (A2D2 visualisation crops carry HUD text / boxes)
    overlay = None
    if args.overlay_mask:
        samp, step = [], max(1, nframes // 30)
        for k in range(0, nframes, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, im = cap.read()
            if ok:
                samp.append(cv2.resize(im, (w, h)))
        overlay = BS.static_overlay_mask(samp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    overlay_frac = float(overlay.mean()) if overlay is not None else 0.0

    # pass 1: models + raw observations
    obs_all, objs_all, has_all, scales, plane_pitches, vp_rows = [], [], [], [], [], []
    rng = np.random.default_rng(0)
    frame_i = 0
    maxf = args.max_frames or 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        objs, lane_boxes, edge_boxes = frame_objects(pack, frame_i, h, w)
        has = objs is not None
        objs = objs or []
        if args.smoke and pack is None:
            z = 18.0 - frame_i * 0.3
            v = cy + fy * cam_h / max(z, 2.0)
            u = cx + fx * 1.8 / max(z, 2.0)
            bw = fx * 1.8 / max(z, 2.0)
            objs = [{"cls": "car", "score": 0.9, "box": [u - bw / 2, v - bw * 0.8, u + bw / 2, v], "b3": None}]
            has = True
        if depth_model is not None:
            import torch

            with torch.inference_mode():
                disp = depth_model.infer_image(rgb)
            depth, s = disparity_to_metric(disp, fy, cy, cam_h, h, w, y0=y0, y1=y1)
            if s is not None:
                scales.append(s)
        else:
            depth = None
        ob = observe_frame(bgr, rgb, predictor, objs, lane_boxes, edge_boxes, h, w, y0, y1, gm0, overlay, cx, use_rails)
        if depth is not None and frame_i % args.plane_every == 0:
            pp = BS.dav2_plane_pitch(depth, ob["road"], fx, fy, cx, cy, (max(y0, cy + 10), y1), rng=rng)
            if pp is not None:
                plane_pitches.append(pp[0])
        vp_rows += ob["vp"]
        ob.pop("road")
        obs_all.append(ob)
        objs_all.append(objs)
        has_all.append(has)
        frame_i += 1
        if maxf and frame_i >= maxf:
            break
        if frame_i % 50 == 0:
            print("pass1", frame_i, flush=True)
    cap.release()
    if frame_i == 0:
        print("ERROR: no frames read", args.video, flush=True)
        sys.exit(3)

    # per-clip ground (median over frames)
    ground = BS.estimate_clip_ground(cam, h, vp_rows, scales, plane_pitches if args.use_plane_pitch else None,
                                     max_pitch_deg=args.max_pitch_deg)
    mode = args.pitch_mode
    if mode == "auto":
        # calibrated rigs (KITTI P2) keep their cy; the smoke run's trapezoid road gives a fake VP
        calibrated = "kitti" in str(cam.get("calib_source", "")).lower()
        mode = "zero" if (calibrated or args.smoke) else "estimate"
    if cam.get("horizon_v") is not None:
        # measured road vanishing row (e.g. Munich: ~552 px, not cy=360) -> equivalent pitch
        mode = "horizon_v"
        p_h = math.atan((cy - float(cam["horizon_v"])) / fy)
        ground["pitch_rad"], ground["pitch_deg"] = p_h, math.degrees(p_h)
    if mode == "zero":
        ground["pitch_rad"], ground["pitch_deg"] = 0.0, 0.0
    ground["pitch_mode"] = mode
    gm = BS.GroundModel(fx, fy, cx, cy, cam_h, ground["pitch_rad"])
    ground["horizon_v"] = gm.horizon_v
    ground["plane_pitch_deg_median"] = float(np.degrees(np.median(plane_pitches))) if plane_pitches else None
    print("ground", json.dumps(ground), flush=True)

    # pass 2a: objects
    trk = BS.ObjTracker(fps, speed, high=args.high, low=args.low, coast=args.coast, min_len=args.min_len,
                        relink_m=args.relink_m, relink_gap=args.coast)
    for k, (objs, has) in enumerate(zip(objs_all, has_all)):
        dets = []
        for o in objs:
            bev, src = obj_bev(o, gm, h, args.obj_range)
            if bev is None or not (0.5 < bev[1] < 80 and abs(bev[0]) < 40):
                continue
            dets.append({"cls": o["cls"], "score": o["score"], "box": o["box"], "bev": bev, "src": src})
        trk.step(k, dets, has_dets=has)
    tracks = trk.finalize()
    by_id = {t["id"]: t for t in tracks}

    # pass 2b: lines
    lt = BS.LineTracker("lane", alpha=args.alpha, gate=args.gate, hold=args.hold, max_tracks=6, motion=args.line_motion)
    ds_line = ds if args.line_motion == "straight" else 0.0
    et = EdgePair(args)
    lanes_live, edges_live, raw_counts = [], [], []
    for k, ob in enumerate(obs_all):
        prior = None
        if lt.tracks:
            good = [t for t in lt.tracks if t["hits"] >= 3]
            if good:
                prior = np.mean([t["c"] for t in good], 0)
        fits, efits, _ = fit_frame_lines(ob, gm, prior, args)
        lanes_live.append(lt.step(k, fits, ds_line))
        edges_live.append(et.step(k, efits, ds_line))
        raw_counts.append({"lane_px": int(len(ob["lane_px"])), "edge_px_L": int(len(ob["edge_px"][0])),
                           "edge_px_R": int(len(ob["edge_px"][1])), "lane_fits": len(fits), "edge_fits": len(efits),
                           "rail_pairs": int(ob["n_rail_pairs"])})
    if args.line_window > 1:
        def recentre(live_list, hist):
            sm = {key: BS.centred_smooth(hh, ds_line, args.line_window) for key, hh in hist.items()}
            out = []
            for k, live in enumerate(live_list):
                rows = []
                for r in live:
                    key = (r["side"], r["id"]) if "side" in r else r["id"]
                    rr = sm.get(key, {}).get(k)
                    rows.append(dict(rr, side=r.get("side")) if rr is not None else r)
                out.append(rows)
            return out
        lanes_live = recentre(lanes_live, lt.hist)
        edges_live = recentre(edges_live, et.hist)

    # pass 3: render (stream) + dumps
    tag_still = Path(args.stills)
    tag_still.mkdir(parents=True, exist_ok=True)
    out_mp4 = Path(args.output)
    sw = StreamWriter(out_mp4, fps, (w + BEV_W, h))
    busy = [sum(1 for t in tracks for r in t["rows"] if r["frame"] == k) * 10 + len(lanes_live[k]) + len(edges_live[k])
            for k in range(frame_i)]
    still_frames = set(pick_still_frames(frame_i, busy))
    still_paths = []
    cap = cv2.VideoCapture(str(args.video))
    poly_frames = []
    for k in range(frame_i):
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h))
        front = draw_front2(bgr, gm, k, tracks, lanes_live[k], edges_live[k], args.front_boxes)
        bev = draw_bev2(h, k, tracks, lanes_live[k], edges_live[k], speed, fps, args.tail)
        im = np.hstack([front, bev])
        sw.write(im)
        if k in still_frames:
            p = tag_still / f"{tag}_f{k:04d}.png"
            cv2.imwrite(str(p), im)
            still_paths.append(str(p))

        def pack_lines(recs):
            outl = []
            for r in recs:
                outl.append({"id": int(r["id"]), "side": r.get("side"), "c": [float(v) for v in r["c"]],
                             "zmin": float(r["zmin"]), "zmax": float(r["zmax"]), "hits": int(r["hits"]),
                             "held": bool(r["held"])})
            return outl

        lb = [BS.sample_curve(r, 1.0, MAX_DIST) for r in lanes_live[k]]
        eb = [BS.sample_curve(r, 1.0, MAX_DIST) for r in edges_live[k]]
        poly_frames.append({
            "frame": k,
            "lanes": pack_lines(lanes_live[k]), "edges": pack_lines(edges_live[k]),
            "lanes_bev": [p.round(3).tolist() for p in lb if len(p) >= 2],
            "edges_bev": [p.round(3).tolist() for p in eb if len(p) >= 2],
            "lanes_px": [np.nan_to_num(gm.project(p)).round(1).tolist() for p in lb if len(p) >= 2],
            "edges_px": [np.nan_to_num(gm.project(p)).round(1).tolist() for p in eb if len(p) >= 2],
            "raw": raw_counts[k],
        })
    cap.release()
    sw.close()
    n_out = count_frames(out_mp4)
    if n_out != frame_i:
        print(f"ERROR: frame count mismatch written {frame_i} readback {n_out}", flush=True)
        sys.exit(4)

    travel = lambda f: speed * f / fps  # noqa: E731
    csv_rows = []
    for t in tracks:
        for r in t["rows"]:
            csv_rows.append({
                "frame": r["frame"], "track_id": t["id"], "class": t["cls"], "score": f"{r['score']:.4f}",
                "bev_x": f"{r['smooth'][0]:.4f}", "bev_z": f"{r['smooth'][1] - travel(r['frame']):.4f}",
                "filt_x": f"{r['filt'][0]:.4f}", "filt_z": f"{r['filt'][1] - travel(r['frame']):.4f}",
                "meas_x": "" if r["meas"] is None else f"{r['meas'][0]:.4f}",
                "meas_z": "" if r["meas"] is None else f"{r['meas'][1] - travel(r['frame']):.4f}",
                "coasted": int(r["coast"]), "src": r["src"], "stationary": int(t["stationary"]),
            })
    csv_rows.sort(key=lambda r: (r["frame"], r["track_id"]))
    tp = Path(args.tracks_csv)
    tp.parent.mkdir(parents=True, exist_ok=True)
    with tp.open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["frame", "track_id", "class", "score", "bev_x", "bev_z", "filt_x", "filt_z",
                                            "meas_x", "meas_z", "coasted", "src", "stationary"])
        wri.writeheader()
        wri.writerows(csv_rows)
    lw = lane_width_check(lanes_live, edges_live)
    # range-scale check: side-parked cars (|x| > 2.5 m) should close at exactly the ego speed.
    # ratio = -dz_meas/dt / speed ~ (fx_true/fx_assumed) * (v_true/v_assumed); 1.0 = consistent.
    ratios = []
    for t in tracks:
        m = [(r["frame"], r["meas"]) for r in t["rows"] if r["meas"] is not None]
        if t["cls"] != "car" or len(m) < 8:
            continue
        fr_ = np.array([a for a, _ in m], float) / fps
        ze = np.array([b[1] for _, b in m]) - speed * fr_
        xs_ = np.array([b[0] for _, b in m])
        if np.median(np.abs(xs_)) < 2.5 or speed <= 0:
            continue
        slope = np.polyfit(fr_, ze, 1)[0]
        ratios.append(float(-slope / speed))
    lw["static_closing_ratio_median"] = float(np.median(ratios)) if ratios else None
    lw["n_side_car_tracks"] = len(ratios)
    pp = Path(args.polylines_json)
    pp.parent.mkdir(parents=True, exist_ok=True)
    pp.write_text(json.dumps({
        "clip": tag, "version": "iter2",
        "units": {"image": "pixels, x right, y down", "bev": "metres in current ego frame, x right, z forward",
                  "c": "x(z) = c0 + c1*z + c2*z^2 in the current ego frame"},
        "K": {"fx": fx, "fy": fy, "cx": cx, "cy": cy}, "cam_height_m": cam_h, "fps": fps, "ego_speed_mps": speed,
        "line_motion": args.line_motion,
        "ground": ground, "lane_width_check": lw, "frames": poly_frames,
    }))
    n_lane_route = sum(1 for o in obs_all if len(o["lane_px"]))
    rec = {
        "rc": 0, "tag": tag, "version": "iter2", "frames_in": frame_i, "frames_out": n_out, "fps": fps,
        "K": [fx, fy, cx, cy], "cam_height": cam_h, "ego_speed_mps": speed, "ground": ground,
        "n_obj_dets_npz": n_obj_dets, "n_tracks": len(tracks), "n_relink": trk.n_relink,
        "n_track_rows": len(csv_rows), "lane_gated": lt.n_gated,
        "edge_gated": sum(t.n_gated for t in et.t.values()), "frames_with_lane_obs": n_lane_route,
        "overlay_mask_frac": overlay_frac, "lane_width_check": lw,
        "stills": still_paths, "tracks_csv": str(tp), "polylines_json": str(pp), "output": str(out_mp4),
        "bytes": out_mp4.stat().st_size, "assumptions": cam.get("note", ""),
    }
    meta_path = Path(args.clip_meta) if args.clip_meta else out_mp4.with_suffix(".json")
    meta_path.write_text(json.dumps(rec, indent=1))
    print("wrote", out_mp4, "frames", frame_i, "tracks", len(tracks), "relinks", trk.n_relink,
          "lane_width", lw, flush=True)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dets", default="")
    p.add_argument("--require-dets", type=int, default=1, help="exit 7 if the WD3D npz is missing/empty")
    p.add_argument("--cam-json", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--speed", type=float, default=12.0)
    p.add_argument("--hfov-deg", type=float, default=0.0, help="override fx from a horizontal FOV (A2D2 check)")
    p.add_argument("--cam-height", type=float, default=0.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--weights", default="weights")
    p.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    p.add_argument("--sam-ckpt", default="sam2.1_hiera_small.pt")
    p.add_argument("--stills", default="stills")
    p.add_argument("--tracks-csv", default="")
    p.add_argument("--polylines-json", default="")
    p.add_argument("--clip-meta", default="")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    # tracking
    p.add_argument("--high", type=float, default=0.5)
    p.add_argument("--low", type=float, default=0.25)
    p.add_argument("--coast", type=int, default=10)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--relink-m", type=float, default=1.5)
    p.add_argument("--obj-range", choices=("fuse", "wd3d", "ground"), default="fuse")
    p.add_argument("--tail", type=int, default=40, help="trajectory tail length in frames")
    # lines
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--gate", type=float, default=0.5)
    p.add_argument("--hold", type=int, default=15)
    p.add_argument("--line-motion", choices=("follow", "straight"), default="follow")
    p.add_argument("--line-window", type=int, default=7, help="centred window over line tracks (1 = causal EMA only)")
    p.add_argument("--min-lane-pts", type=int, default=12)
    p.add_argument("--max-line-z", type=float, default=40.0)
    p.add_argument("--overlay-mask", type=int, default=1)
    p.add_argument("--use-plane-pitch", type=int, default=1)
    p.add_argument("--pitch-mode", choices=("auto", "zero", "estimate"), default="auto")
    p.add_argument("--max-pitch-deg", type=float, default=3.0)
    p.add_argument("--rails", type=int, default=-1, help="-1 = from cams json 'rails' (default off)")
    p.add_argument("--plane-every", type=int, default=5)
    p.add_argument("--front-boxes", type=int, default=1)
    args = p.parse_args()
    tag = args.tag or Path(args.video).stem
    if not args.tracks_csv:
        args.tracks_csv = str(Path("tracks") / f"{tag}_tracks.csv")
    if not args.polylines_json:
        args.polylines_json = str(Path("polylines") / f"{tag}_polylines.json")
    if args.smoke:
        args.device = "cpu"
        if not args.dets:
            args.require_dets = 0
    run_clip2(args)


if __name__ == "__main__":
    main()
