#!/usr/bin/env python3
"""Foundation-model BEV: WildDet3D + SAM 2.1 hiera-small + Depth Anything V2 Small.

No TwinLiteNet, no YOLO, no Grounding DINO, no SAM 3.
Front view: class-coloured boxes + lane/edge polylines.
BEV: ego path, object trajectories, lane-line curves, road-edge curves only.
Depth is used only to lift polylines and box bottoms onto the ground plane.
"""
from __future__ import annotations

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


def lane_polylines(mask, cy, height, min_run=2, max_run=48):
    """Centre of each thin mask run, linked down the image. From front-bev-tracks overlay.py."""
    if mask is None:
        return []
    y0 = max(int(cy) + 6, 0)
    row_index = range(y0, height - 8, 3)
    rows = []
    for v in row_index:
        cols = np.flatnonzero(mask[v] > 0)
        if len(cols) == 0:
            rows.append([])
            continue
        splits = np.where(np.diff(cols) > 6)[0] + 1
        points = []
        for run in np.split(cols, splits):
            if not (min_run <= len(run) <= max_run):
                continue
            points.append((float(run.mean()), float(v)))
        rows.append(points)
    used = [set() for _ in rows]
    lines = []
    for r, points in enumerate(rows):
        for i, point in enumerate(points):
            if i in used[r]:
                continue
            chain = [point]
            used[r].add(i)
            prev = point
            misses = 0
            for n in range(r + 1, min(r + 120, len(rows))):
                if not rows[n]:
                    misses += 1
                    if misses > 2:
                        break
                    continue
                misses = 0
                best, best_d = None, 48.0
                for j, cand in enumerate(rows[n]):
                    if j in used[n]:
                        continue
                    dist = abs(cand[0] - prev[0])
                    if dist < best_d:
                        best, best_d = j, dist
                if best is None:
                    break
                used[n].add(best)
                prev = rows[n][best]
                chain.append(prev)
            chain = np.asarray(chain, dtype=np.float64)
            if len(chain) >= 6 and np.linalg.norm(np.diff(chain, axis=0), axis=1).sum() >= 30.0:
                lines.append(chain[::2])
    return lines


def edge_polylines_from_road(road, y0, y1, min_len=12):
    """Left and right boundaries of a road-surface mask, in the road band."""
    h, w = road.shape
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    left, right = [], []
    for v in range(y0, y1):
        cols = np.flatnonzero(road[v] > 0)
        if cols.size < 8:
            continue
        left.append((float(cols[0]), float(v)))
        right.append((float(cols[-1]), float(v)))
    out = []
    for pts in (left, right):
        if len(pts) >= min_len:
            arr = np.asarray(pts, np.float64)
            arr = arr[::2]
            if len(arr) >= 6:
                out.append(arr)
    return out


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


def bright_lane_mask(bgr, road, y0, y1):
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 90)
    yellow = (
        (hsv[:, :, 0] >= 12)
        & (hsv[:, :, 0] <= 42)
        & (hsv[:, :, 1] > 50)
        & (hsv[:, :, 2] > 70)
    )
    # stay inside the road, away from left/right edges (guardrails, kerbs)
    if road.dtype != np.uint8:
        road_u = (road > 0).astype(np.uint8)
    else:
        road_u = (road > 0).astype(np.uint8)
    k = max(9, w // 40)
    road_inner = cv2.erode(road_u, np.ones((1, k), np.uint8), iterations=2)
    m = (white | yellow) & (road_inner > 0)
    m[: int(y0)] = False
    m[int(y1) :] = False
    m = (m.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, num):
        x, y, ww, hh, area = st[i]
        if area < 18 or area > 0.05 * h * w:
            continue
        if ww > 90 and hh > 90:
            continue
        if max(ww, hh) < 12:
            continue
        out[lab == i] = 255
    return out


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


def smooth_poly(pts, stride=1):
    if pts is None or len(pts) < 3:
        return pts
    arr = np.asarray(pts, np.float64)
    k = min(5, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    if k < 3:
        return arr
    ker = np.ones(k) / k
    xs = np.convolve(arr[:, 0], ker, mode="same")
    ys = np.convolve(arr[:, 1], ker, mode="same")
    # keep endpoints
    xs[0], ys[0] = arr[0]
    xs[-1], ys[-1] = arr[-1]
    out = np.stack([xs, ys], axis=1)
    return out[::stride] if stride > 1 else out


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


class Tracker:
    def __init__(self, max_dist=12.0, max_age=8):
        self.max_dist = max_dist
        self.max_age = max_age
        self.next_id = 1
        self.tracks = []  # dicts

    def update(self, frame, dets):
        """dets: list of dicts with cls, score, box, bev (x,z), xyz or None."""
        used_t, used_d = set(), set()
        pairs = []
        for ti, tr in enumerate(self.tracks):
            if frame - tr["last"] > self.max_age:
                continue
            for di, d in enumerate(dets):
                if d["cls"] != tr["cls"]:
                    continue
                iou = iou_xyxy(tr["box"], d["box"]) if tr.get("box") is not None else 0.0
                dist = math.hypot(tr["bev"][0] - d["bev"][0], tr["bev"][1] - d["bev"][1])
                if iou < 0.25 and dist > self.max_dist:
                    continue
                cost = dist - 8.0 * iou
                pairs.append((cost, ti, di))
        pairs.sort()
        for cost, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            d = dets[di]
            tr = self.tracks[ti]
            tr["box"] = d["box"]
            tr["bev"] = d["bev"]
            tr["score"] = d["score"]
            tr["xyz"] = d.get("xyz")
            tr["last"] = frame
            tr["hits"] += 1
            tr["hist"].append(
                {"frame": frame, "bev": d["bev"], "score": d["score"], "xyz": d.get("xyz")}
            )
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            self.tracks.append(
                {
                    "id": self.next_id,
                    "cls": d["cls"],
                    "box": d["box"],
                    "bev": d["bev"],
                    "score": d["score"],
                    "xyz": d.get("xyz"),
                    "last": frame,
                    "hits": 1,
                    "hist": [{"frame": frame, "bev": d["bev"], "score": d["score"], "xyz": d.get("xyz")}],
                }
            )
            self.next_id += 1
        self.tracks = [t for t in self.tracks if frame - t["last"] <= self.max_age]
        live = [t for t in self.tracks if t["last"] == frame]
        return live


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


def draw_bev(h, tracks, lanes_bev, edges_bev, frame, speed, fps, bev_w=BEV_W):
    panel = np.full((h, bev_w, 3), 22, np.uint8)
    scale = bev_scale(h, bev_w)
    # tiny distance ticks only (not a depth map)
    for metre in (10, 20, 30, 40):
        y = int(round(h - 20 - metre * scale))
        if 0 <= y < h:
            cv2.line(panel, (0, y), (10, y), (80, 80, 80), 1)
            cv2.putText(panel, f"{metre}m", (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1, cv2.LINE_AA)

    def paint_curves(polys, color, width=2):
        for poly in polys:
            pts = []
            for xz in poly:
                if xz is None or not np.isfinite(xz).all():
                    continue
                u, v = to_bev_px(xz, h, bev_w, scale)
                if -30 <= u < bev_w + 30 and -30 <= v < h + 30:
                    pts.append((u, v))
            if len(pts) >= 2:
                cv2.polylines(panel, [np.asarray(pts, np.int32)], False, color, width, cv2.LINE_AA)

    paint_curves(edges_bev, COL_EDGE, 2)
    paint_curves(lanes_bev, COL_LANE, 2)

    ds = float(speed) / max(float(fps), 1e-6)
    for tr in tracks:
        hist = [s for s in tr["hist"] if s["frame"] <= frame][-50:]
        trail = []
        for s in hist:
            z = s["bev"][1] - (frame - s["frame"]) * ds
            x = s["bev"][0]
            u, v = to_bev_px((x, z), h, bev_w, scale)
            trail.append((u, v))
        col = COL_CLASS.get(tr["cls"], (200, 200, 200))
        if len(trail) >= 2:
            cv2.polylines(panel, [np.asarray(trail, np.int32)], False, col, 2, cv2.LINE_AA)
        if tr["last"] == frame and trail:
            u, v = trail[-1]
            if 0 <= u < bev_w and 0 <= v < h:
                cv2.circle(panel, (u, v), 5, col, -1, cv2.LINE_AA)
                cv2.circle(panel, (u, v), 5, (0, 0, 0), 1, cv2.LINE_AA)

    # ego path (straight, constant heading) and marker at origin
    path = []
    for k in range(0, 80):
        u, v = to_bev_px((0.0, -k * ds), h, bev_w, scale)
        if 0 <= v < h:
            path.append((u, v))
    if len(path) >= 2:
        cv2.polylines(panel, [np.asarray(path, np.int32)], False, COL_PATH, 2, cv2.LINE_AA)
    nose = to_bev_px((0.0, 2.0), h, bev_w, scale)
    left = to_bev_px((-0.9, -0.5), h, bev_w, scale)
    right = to_bev_px((0.9, -0.5), h, bev_w, scale)
    cv2.fillConvexPoly(panel, np.array([nose, left, right], np.int32), COL_EGO)
    return panel


def draw_front(frame, live, lanes_px, edges_px):
    img = frame.copy()
    for poly in edges_px:
        pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, COL_EDGE, 2, cv2.LINE_AA)
    for poly in lanes_px:
        pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, COL_LANE, 2, cv2.LINE_AA)
    for tr in live:
        x1, y1, x2, y2 = [int(round(v)) for v in tr["box"]]
        col = COL_CLASS.get(tr["cls"], (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f"{tr['cls']}#{tr['id']}"
        cv2.putText(img, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return img


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


def dets_for_frame(pack, frame):
    if pack is None:
        return None
    if frame in pack["by"]:
        return pack["by"][frame]
    prev = [k for k in pack["by"] if k <= frame]
    if not prev:
        return None
    return pack["by"][max(prev)]


def split_dets(pack, frame, h, w, y0, y1, cam_height, depth, fx, fy, cx, cy, score_min=0.25):
    raw = dets_for_frame(pack, frame) if pack else None
    objs, lane_boxes, edge_boxes = [], [], []
    if raw is None:
        return objs, lane_boxes, edge_boxes, {}
    prompts = pack["prompts"]
    stats = defaultdict(int)
    for i, cid in enumerate(raw["cids"]):
        cid = int(cid)
        name = prompts[cid] if 0 <= cid < len(prompts) else ""
        sc = float(raw["scores"][i])
        box = raw["boxes"][i]
        stats[name] += 1
        if name in LANE_PROMPTS and useful_prompt_box(box, h, w, y0, y1):
            lane_boxes.append(box)
        elif name in EDGE_PROMPTS and useful_prompt_box(box, h, w, y0, y1):
            edge_boxes.append(box)
        if name not in OBJ_PROMPTS or sc < score_min:
            continue
        xyz = None
        bev = None
        got = sane_wd3d_xz(raw["boxes3d"][i], cam_height)
        if got is not None:
            bev = (got[0], got[1])
            xyz = (got[2], got[3], got[4])
        if bev is None:
            u = 0.5 * (float(box[0]) + float(box[2]))
            v = min(float(box[3]), h - 2)
            lifted = lift_uv([u], [v], depth, fx, fy, cx, cy)[0]
            if np.isfinite(lifted).all():
                bev = (float(lifted[0]), float(lifted[1]))
        if bev is None:
            continue
        objs.append({"cls": name, "score": sc, "box": [float(v) for v in box], "bev": bev, "xyz": xyz})
    return objs, lane_boxes, edge_boxes, dict(stats)



def filter_lane_polys(lines, w, h, y0, y1):
    """Drop short, horizontal, or far-side polylines that are usually guardrails or wires."""
    kept = []
    for p in lines:
        if p is None or len(p) < 4:
            continue
        arr = np.asarray(p, np.float64)
        xs, ys = arr[:, 0], arr[:, 1]
        if ys.max() - ys.min() < max(36.0, 0.08 * h):
            continue
        if xs.mean() < 0.12 * w or xs.mean() > 0.88 * w:
            continue
        # lanes run up the image; reject mostly-horizontal strokes
        i_lo, i_hi = int(np.argmax(ys)), int(np.argmin(ys))
        dx, dy = abs(xs[i_hi] - xs[i_lo]), abs(ys[i_hi] - ys[i_lo])
        if dy < 0.45 * dx:
            continue
        kept.append(arr)
    return kept


def extract_lines(route, predictor, bgr, rgb, lane_boxes, edge_boxes, h, w, y0, y1, cy):
    lanes_px, edges_px = [], []
    road = None
    if predictor is not None:
        predictor.set_image(rgb)
    if route in ("wd3d_box_prompts", "hybrid"):
        if predictor is not None and lane_boxes:
            um = union_masks(sam_masks_from_boxes(predictor, lane_boxes, h, w, y0, y1), h, w)
            lanes_px = lane_polylines(um, cy=min(cy, y0), height=h)
        if predictor is not None and edge_boxes:
            um = union_masks(sam_masks_from_boxes(predictor, edge_boxes, h, w, y0, y1), h, w)
            edges_px = lane_polylines(um, cy=min(cy, y0), height=h, min_run=2, max_run=80)
            if len(edges_px) < 1:
                edges_px = edge_polylines_from_road(um, y0, y1)
    need_fb = route in ("fallback_lower_half", "hybrid") or (not lanes_px or not edges_px)
    if need_fb:
        if predictor is not None:
            road = sam_road_mask(predictor, h, w, y0, y1)
        else:
            road = trapezoid_road(h, w, y0, y1)
        if not edges_px:
            edges_px = edge_polylines_from_road(road, y0, y1)
        if not lanes_px:
            lm = bright_lane_mask(bgr, road, y0, y1)
            lanes_px = lane_polylines(lm, cy=min(cy, y0), height=h)
    lanes_px = [smooth_poly(p) for p in lanes_px]
    edges_px = [smooth_poly(p) for p in edges_px]
    lanes_px = filter_lane_polys(lanes_px, w, h, y0, y1)
    return lanes_px, edges_px


def decide_route(sample_hits, n_sample, min_frac=0.25):
    if n_sample <= 0:
        return "fallback_lower_half"
    frac = sample_hits / n_sample
    if frac >= min_frac:
        return "wd3d_box_prompts"
    if sample_hits > 0:
        return "hybrid"
    return "fallback_lower_half"


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


def poly_to_list(polys):
    out = []
    for p in polys:
        arr = np.asarray(p, np.float32)
        out.append([[float(a[0]), float(a[1])] for a in arr if np.isfinite(a).all()])
    return [p for p in out if len(p) >= 2]


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


def run_clip(args):
    cam = json.loads(Path(args.cam_json).read_text()) if args.cam_json else {}
    if args.tag and args.tag in cam:
        cam = cam[args.tag]
    elif "fx" not in cam and len(cam) == 1:
        cam = next(iter(cam.values()))
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    cam_h = float(cam.get("height", 1.5))
    speed = float(args.speed)
    road_y0f = float(cam.get("road_y0", 0.4))
    road_y1f = float(cam.get("road_y1", 0.98))

    pack = load_dets(args.dets)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y0, y1 = road_y0f * h, road_y1f * h
    out_w = w + BEV_W

    device = args.device
    if args.smoke or device.startswith("cpu"):
        import os

        if not args.smoke and device.startswith("cuda"):
            pass
    depth_model = None
    predictor = None
    if not args.smoke:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available, abort (not a smoke run)", flush=True)
            sys.exit(2)
        depth_model = load_depth_anything(device, Path(args.weights))
        predictor = load_sam_predictor(
            device, Path(args.weights), args.sam_config, args.sam_ckpt
        )

    # peek a few frames with dets to decide prompting route
    sample_hits, n_sample = 0, 0
    if pack is not None and not args.force_route:
        peek = 0
        tmp_cap = cv2.VideoCapture(str(args.video))
        i = 0
        while peek < 12:
            ok, _ = tmp_cap.read()
            if not ok:
                break
            raw = dets_for_frame(pack, i)
            if raw is not None:
                n_sample += 1
                _, lb, eb, _ = split_dets(pack, i, h, w, y0, y1, cam_h, np.ones((h, w), np.float32) * 10, fx, fy, cx, cy)
                if lb or eb:
                    sample_hits += 1
                peek += 1
            i += 1
            if i > 80:
                break
        tmp_cap.release()
    route = args.force_route or decide_route(sample_hits, n_sample)
    print(f"prompt_route={route} sample_hits={sample_hits}/{n_sample}", flush=True)

    tracker = Tracker(max_dist=15.0 if speed >= 18 else 10.0)
    frames_out = []
    csv_rows = []
    poly_frames = []
    busy = []
    scales = []
    frame_i = 0
    maxf = args.max_frames or 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if depth_model is not None:
            import torch

            with torch.inference_mode():
                disp = depth_model.infer_image(rgb)
            depth, s = disparity_to_metric(disp, fy, cy, cam_h, h, w, y0=y0, y1=y1)
            if depth is None:
                depth = flat_ground_depth(h, w, fy, cy, cam_h)
            else:
                scales.append(s)
        else:
            depth = flat_ground_depth(h, w, fy, cy, cam_h)

        objs, lane_boxes, edge_boxes, _ = split_dets(
            pack, frame_i, h, w, y0, y1, cam_h, depth, fx, fy, cx, cy
        )
        if args.smoke and not objs:
            # one synthetic car so the CSV path is exercised
            z = 15.0 - frame_i * 0.4
            objs = [{"cls": "car", "score": 0.9, "box": [w * 0.4, h * 0.5, w * 0.55, h * 0.72], "bev": (1.5, z), "xyz": (1.5, cam_h, z)}]

        lanes_px, edges_px = extract_lines(route, predictor, bgr, rgb, lane_boxes, edge_boxes, h, w, y0, y1, cy)
        lanes_bev, edges_bev = [], []
        for poly in lanes_px:
            xz = lift_uv(poly[:, 0], poly[:, 1], depth, fx, fy, cx, cy)
            lanes_bev.append(xz)
        for poly in edges_px:
            xz = lift_uv(poly[:, 0], poly[:, 1], depth, fx, fy, cx, cy)
            edges_bev.append(xz)

        live = tracker.update(frame_i, objs)
        front = draw_front(bgr, live, lanes_px, edges_px)
        bev = draw_bev(h, tracker.tracks, lanes_bev, edges_bev, frame_i, speed, fps)
        if bev.shape[:2] != (h, BEV_W):
            bev = cv2.resize(bev, (BEV_W, h), interpolation=cv2.INTER_NEAREST)
        combined = np.hstack([front, bev])
        assert combined.shape == (h, out_w, 3), combined.shape
        frames_out.append(combined)

        poly_len = sum(len(p) for p in lanes_px) + sum(len(p) for p in edges_px)
        busy.append(len(live) * 10 + poly_len)
        for tr in live:
            x = y = z = ""
            if tr.get("xyz"):
                x, y, z = (f"{v:.4f}" for v in tr["xyz"])
            csv_rows.append(
                {
                    "frame": frame_i,
                    "track_id": tr["id"],
                    "class": tr["cls"],
                    "score": f"{tr['score']:.4f}",
                    "bev_x": f"{tr['bev'][0]:.4f}",
                    "bev_z": f"{tr['bev'][1]:.4f}",
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
        poly_frames.append(
            {
                "frame": frame_i,
                "lanes_px": poly_to_list(lanes_px),
                "edges_px": poly_to_list(edges_px),
                "lanes_bev": poly_to_list(lanes_bev),
                "edges_bev": poly_to_list(edges_bev),
            }
        )
        frame_i += 1
        if maxf and frame_i >= maxf:
            break
        if frame_i % 30 == 0:
            print(frame_i, flush=True)

    cap.release()
    if frame_i == 0:
        print("ERROR: no frames read", args.video, flush=True)
        sys.exit(3)

    out_mp4 = Path(args.output)
    write_video(out_mp4, frames_out, fps, (out_w, h))
    n_out = count_frames(out_mp4)
    if n_out != frame_i:
        print(f"ERROR: frame count mismatch written {frame_i} readback {n_out}", flush=True)
        sys.exit(4)

    tag = args.tag or Path(args.video).stem
    still_dir = Path(args.stills)
    still_dir.mkdir(parents=True, exist_ok=True)
    still_paths = []
    for fi in pick_still_frames(frame_i, busy):
        p = still_dir / f"{tag}_f{fi:04d}.png"
        cv2.imwrite(str(p), frames_out[fi])
        still_paths.append(str(p))

    tracks_path = Path(args.tracks_csv)
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    with tracks_path.open("w", newline="") as f:
        wri = csv.DictWriter(
            f, fieldnames=["frame", "track_id", "class", "score", "bev_x", "bev_z", "x", "y", "z"]
        )
        wri.writeheader()
        wri.writerows(csv_rows)

    poly_path = Path(args.polylines_json)
    poly_path.parent.mkdir(parents=True, exist_ok=True)
    poly_path.write_text(
        json.dumps(
            {
                "clip": tag,
                "units": {
                    "image": "pixels, x right, y down",
                    "bev": "metres in current ego frame, x right, z forward",
                },
                "K": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "cam_height_m": cam_h,
                "prompt_route": route,
                "frames": poly_frames,
            },
            indent=1,
        )
    )

    rec = {
        "rc": 0,
        "tag": tag,
        "prompt_route": route,
        "sample_hits": sample_hits,
        "n_sample": n_sample,
        "frames_in": frame_i,
        "frames_out": n_out,
        "fps": fps,
        "K": [fx, fy, cx, cy],
        "cam_height": cam_h,
        "ego_speed_mps": speed,
        "depth_scale_median": (float(np.median(scales)) if scales else None),
        "n_track_rows": len(csv_rows),
        "stills": still_paths,
        "tracks_csv": str(tracks_path),
        "polylines_json": str(poly_path),
        "output": str(out_mp4),
        "bytes": out_mp4.stat().st_size,
        "assumptions": cam.get("note", ""),
    }
    meta_path = Path(args.clip_meta) if args.clip_meta else out_mp4.with_suffix(".json")
    meta_path.write_text(json.dumps(rec, indent=1))
    print("wrote", out_mp4, "frames", frame_i, "route", route, flush=True)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dets", default="")
    p.add_argument("--cam-json", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--speed", type=float, default=12.0)
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
    p.add_argument("--force-route", default="")
    args = p.parse_args()
    tag = args.tag or Path(args.video).stem
    if not args.tracks_csv:
        args.tracks_csv = str(Path("tracks") / f"{tag}_tracks.csv")
    if not args.polylines_json:
        args.polylines_json = str(Path("polylines") / f"{tag}_polylines.json")
    if args.smoke:
        args.device = "cpu"
        args.force_route = args.force_route or "fallback_lower_half"
    run_clip(args)


if __name__ == "__main__":
    main()
