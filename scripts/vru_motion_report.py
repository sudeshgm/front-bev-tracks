#!/usr/bin/env python3
"""Per-track VRU speeds for one clip: raw vs smooth_world (old) vs VRU smoother (new).

Needs only what overlay.py --rfdetr --rfdetr-mode label already cached: no
frames, no GPU. Run it on the machine that holds the clips, once per clip:

    python scripts/vru_motion_report.py --camera CAM.json --motion CLIP_motion.csv \
        --cache CLIP.npz [--rfdetr-cache CLIP_rfdetr.npz] [--window T0 T1] --clip vru058 \
        --json out/vru058_vru_motion.json

--window is the tag window in clip seconds (turn +- 10 s); default the whole clip.
Speeds: raw = GLS straight-line fit of the raw world positions (ego pose +
ground contact); old = median step speed of the smooth_world positions the
tags used to read; new = median speed of the VRU smoother. "stationary" is
the VRU smoother's standing test (net displacement < 2x expected noise).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import overlay as ov  # noqa: E402
from qdtrack import vru_motion  # noqa: E402


def _speed_of(times, points):
    if len(points) < 2:
        return float("nan")
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.median(step / np.maximum(np.diff(times), 1e-3)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="YOLOPv2 detection cache (.npz) from overlay.py")
    parser.add_argument("--rfdetr-cache", type=Path, help="Default: <cache>_rfdetr.npz")
    parser.add_argument("--rfdetr-threshold", type=float, default=0.5)
    parser.add_argument("--window", type=float, nargs=2, metavar=("T0", "T1"))
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--clip", default="")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    camera = ov.CylinderCamera.load(args.camera)
    samples, poses, frame_dt = ov.load_motion(args.motion)
    rf_mod = ov._rfdetr()
    rf_cache = args.rfdetr_cache or rf_mod.default_cache(args.cache)
    rows, names = rf_mod.load_cache(rf_cache)
    if len(rows) != len(samples):
        raise SystemExit(f"RF-DETR cache has {len(rows)} frames, motion file has {len(samples)}")
    rf = {"rows": rows, "names": names, "threshold": args.rfdetr_threshold, "mode": "label"}
    ov.VRU_SMOOTHER = "legacy"  # keep smooth_world in "world"; the VRU result is in world_vru
    _vehicles, others, _packed, _render = ov.track_label_mode(samples, poses, args.cache, camera, rf, frame_dt)
    t0, t1 = (args.window if args.window else (None, None))
    report = []
    for track in others:
        if not vru_motion.is_vru(track.get("kind")) or track.get("hits", 0) < args.min_hits:
            continue
        motion = vru_motion.smooth_vru_track(track, poses, frame_dt)
        mask = motion.window_mask(t0, t1)
        if mask.sum() < 2:
            continue
        measured = [s for s in track["samples"] if not s.get("predicted")]
        times = np.array([poses[int(s["token"])]["t"] for s in measured])
        order = np.argsort(times)
        old = np.array([s["world"] for s in measured], dtype=np.float64)[order]
        tsorted = times[order]
        wmask = np.ones(len(tsorted), dtype=bool)
        if t0 is not None:
            wmask &= (tsorted >= t0 - 1e-9) & (tsorted <= t1 + 1e-9)
        idx = np.flatnonzero(mask)
        stat = motion.stationary_report(t0, t1)
        raw_disp = motion._fitted_displacement(idx, vru_motion.DEFAULT_CONFIG)
        report.append({
            "clip": args.clip, "id": int(track["id"]), "kind": track["kind"], "samples": int(mask.sum()),
            "range_m": round(float(np.median(motion.ranges[mask])), 1),
            "raw_speed": round(float(np.linalg.norm(raw_disp)) / max(stat["duration"], 1e-3), 2),
            "old_speed": round(_speed_of(tsorted[wmask], old[wmask]), 2),
            "new_speed": round(motion.speed(t0, t1), 2),
            "stationary": bool(stat["stationary"]),
            "displacement_m": round(stat["displacement"], 2), "noise_m": round(stat["noise"], 2),
            "outliers": int((~motion.inlier[mask]).sum()),
        })
    print("| clip | id | kind | n | range m | raw m/s | old m/s | new m/s | standing | disp m | 2x noise m | outliers |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in report:
        print(f"| {r['clip']} | {r['id']} | {r['kind']} | {r['samples']} | {r['range_m']} | {r['raw_speed']} | "
              f"{r['old_speed']} | {r['new_speed']} | {'yes' if r['stationary'] else 'no'} | {r['displacement_m']} | "
              f"{round(2 * r['noise_m'], 2)} | {r['outliers']} |")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
