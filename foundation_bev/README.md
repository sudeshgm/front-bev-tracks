# Foundation-model BEV pipeline

This directory is the foundation-model bird's-eye-view tracker. It is a separate line of work from the YOLO / TwinLiteNet pipeline on `lane-solver-fast` and `main`. The latest GPU-validated code is **iter7**. Earlier iteration specs are under `history/`, and small result tables are under `results/`.

Videos are not in this repo. They live in the private Google Drive folder [KITTI_BEV_Test_Videos](https://drive.google.com/drive/folders/1WlUGEUkBr-VqJg3jSeieE6nXRYUsdCq1). Per-iteration file IDs are in `results/iterN/DRIVE_LINKS.txt`.

## Pipeline

The run is two stages, then a BEV render. TwinLiteNet and YOLO are never used inside the foundation pipeline. YOLOPv2 is an optional measurement-only baseline (`yolo_baseline/`).

1. **WildDet3D** (stage 1) — open-vocabulary 3D boxes for `car`, `person`, `bicycle`, plus lane-line and road-edge prompt boxes.
2. **SAM 2.1** — box-prompted masks (lower-half fallback if a box is missing).
3. **Depth Anything V2** — per-pixel depth, lifted onto a ground-plane BEV.
4. **Tracking and lines**
   - ByteTrack-style association with a Kalman filter.
   - Ego-relative trails, capped at 2 s / 15 m. Cars are drawn red, people and bicycles in their own colours.
   - Near-stationary cars are pinned as a single dot with no trail; young tracks (< 5 rows) and tracks moving more sideways than forward are drawn as a dot at their current position (drawing only; tracks and metrics are unchanged).
   - `lines_mt` tracks lane lines and road edges separately, with split/merge links, fork events, and gore (merge-area) polygons.
5. **Ego speed** — selection order: OXTS, then BEV phase correlation, then Lucas–Kanade flow, then a scene closing-rate, then the per-clip prior. Metrics use each clip's capture rate (`capture_hz`): KITTI 10 Hz, Munich 30 Hz, A2D2 10 Hz.

The BEV overlay shows only class-coloured trajectories, lane-line curves, road-edge curves, the ego path, and gore polygons. It does not show depth maps or edge maps.

## How to run iter7

Entry point: `src/run_iter7.py`. The dataset directory (`FBEV_DS`) must contain the clip MP4s, `cams.json` (intrinsics used at capture time), `motion_assumptions.json`, and this package (`src/`, `cams_iter7.json`, and optionally `yolo_baseline/`). The nine clips are `kitti_highway_sample`, `kitti_highway_exit_approach`, `kitti_highway_late`, `kitti_highway_final`, `kitti_city_cyclist_candidate`, `kitti_lane_split_candidate`, `munich`, `a2d2a`, and `a2d2b`.

Typical GPU run (one T4, about 75 minutes for stage 1 + stage 2 + the YOLO baseline):

```bash
export FBEV_DS=/path/to/dataset
export FBEV_OUT=/path/to/output
export FBEV_TMP=/path/to/scratch
# Optional YOLOPv2 baseline (measurement only):
# export FBEV_YOLOPV2=/path/to/yolopv2.torchscript
# export FBEV_YOLO_REPO=/path/to/sam-bev-22nov
python src/run_iter7.py
```

Outputs land in `FBEV_OUT`: `results/<clip>_foundation_bev_iter7.mp4`, `stills/`, `tracks/`, `polylines/`, `dets_dump/`, `metrics.md`, `ego_speed_table.md`, and `metrics_yolo_vs_run.md`.

### Environment variables

| Variable | Default | Role |
|---|---|---|
| `FBEV_DS` | (auto: Kaggle input, or a local fallback) | Dataset root |
| `FBEV_OUT` | `/kaggle/working` | Output directory |
| `FBEV_TMP` | `/kaggle/temp/w` | Scratch / weights / WildDet3D checkout |
| `FBEV_TAGS` | all 9 clips | Comma-separated clip list |
| `FBEV_REUSE_DETS` | unset | Directory of stage-1 `wd3d_<tag>.npz` or `<tag>_dets_ego.json`; skips WildDet3D and stage 1 |
| `FBEV_SMOKE` | unset | `1` for a short CPU/GPU smoke run |
| `FBEV_MAXF` | `0` (no cap) | Max frames per clip |
| `FBEV_BUDGET_S1` | `2.6 * 3600` | Stage-1 time budget, seconds |
| `FBEV_WALL_CAP` | `3.8 * 3600` | Overall wall-clock cap, seconds |
| `FBEV_EXTRA_STILLS` | built-in set | Comma-separated frame indices to dump |
| `FBEV_YOLOPV2` + `FBEV_YOLO_REPO` | unset | Enable the YOLOPv2 baseline |
| `FBEV_EDGE_MAX_DEG` | `25` | Trim the near-ego part (z ≤ 20 m) of a main road edge whose local heading is more than this many degrees off the road heading; parts that cross a lane are trimmed too, and a 5-frame hold fills the gap |
| `FBEV_LANE_EDGE_DUP_M` | `0.8` | Drop a lane line (not the nearest one per side) that runs within this distance of a main edge over ≥ 70 % of their common span |
| `FBEV_RAY_R` | unset (off) | Experimental ray-aligned measurement noise for off-axis objects; not used in GPU runs |
| `FBEV_GORE_GROW_M` | `1.2` | Widen-gore: gap must rise this many metres above the recent minimum |
| `FBEV_GORE_MAX_STEP` | `0.4` | Widen-gore: max gap growth per step (a jump restarts the history) |

CPU tests (no GPU, no weights):

```bash
pytest -q tests
PYTHONPATH=tests/np1_sim pytest -q tests   # NumPy 1.x compatibility (`np.trapz` fallback)
```

## Changelog

Accel is RMS BEV acceleration of drawn tracks, in m/s². Lane figures are left/right presence at z = 10 m, in percent of frames. Iter2 and iter3 scored KITTI at the stored 4 fps, which understated acceleration; iter4 onward uses `capture_hz` (KITTI 10 Hz). Munich (30 Hz) and A2D2 (10 Hz) are consistent across iterations. Iter5 is the merged 9-clip result (4 highway clips from the first iter5 run + 5 clips recovered in iter5b after a NumPy 1.x `np.trapezoid` crash). Full tables are in `results/iterN/`.

### Accel RMS (m/s²)

| clip | iter2 | iter3 | iter4 | iter5 | iter6 | iter7 |
|---|---:|---:|---:|---:|---:|---:|
| highway sample | 0.504† | 0.584† | 0.108 | 0.095 | 0.095 | 0.095 |
| exit approach | 0.628† | 0.679† | 0.541 | 0.546 | 0.546 | 0.546 |
| highway late | 1.697† | 1.827† | 0.651 | 0.575 | 0.575 | **0.526** |
| highway final | 0.838† | 0.915† | 0.626 | 0.626 | 0.638 | 0.614 |
| city cyclist | 0.474† | 0.585† | 0.566 | 0.566 | 0.566 | 0.566 |
| lane split | 1.131† | 1.134† | 1.052 | 1.074 | **0.997** | 1.048 |
| munich | 1.079 | 0.911 | 0.917 | 0.911 | 0.912 | **0.676** |
| a2d2a | 1.261 | 0.820 | 0.774 | 0.791 | 0.901 | **0.775** |
| a2d2b | 1.119 | 0.890 | 0.890 | 0.890 | **0.827** | 0.879 |

† KITTI scored at 4 fps (not comparable to iter4+).

### Lane presence L10 / R10 (%)

| clip | iter4 | iter5 | iter6 | iter7 |
|---|---:|---:|---:|---:|
| highway sample | 95 / 95 | 92.5 / 92.5 | 95 / 95 | 95 / 95 |
| exit approach | 95 / 90 | 92.5 / 92.5 | 95 / 95 | 95 / 95 |
| highway late | 92.5 / 95 | 87.5 / 92.5 | 92.5 / 95 | 92.5 / 95 |
| highway final | 93.3 / 90 | 83.3 / 86.7 | 93.3 / 90 | 93.3 / 90 |
| city cyclist | 93.3 / 93.3 | 90 / 90 | 93.3 / 93.3 | 93.3 / 93.3 |
| lane split | 93.3 / 76.7 | 90 / 46.7 | 86.7 / 66.7 | **93.3 / 73.3** |
| munich | 87.6 / 65.4 | 84.9 / 56.9 | 85.7 / 65.3 | 85.7 / 65.3 |

### Road-edge presence L10 / R10 (%)

| clip | iter5 | iter6 | iter7 |
|---|---:|---:|---:|
| highway late | 95 / 87.5 | 95 / 77.5 | 95 / 87.5 |
| lane split | 93.3 / 93.3 | 40 / 83.3 | 93.3 / 83.3 |
| munich | 99.2 / 92.4 | 87.9 / 64.0 | 93.9 / 72.7 |
| a2d2a | 79.7 / 77.3 | 76.2 / 71.5 | 96.5 / 94.2 |
| a2d2b | 97.1 / 97.1 | 85.5 / 83.7 | 97.1 / 95.9 |

Other clips are unchanged across iter5–iter7 (sample, exit approach 95 / 95; final 90 / 86.7; city 93.3 / 93.3).

### What changed

| iter | Highlights |
|---|---|
| 1 | First foundation BEV run: WildDet3D + SAM 2.1 + Depth Anything, ground-plane BEV. |
| 2 | Smoother tracks and polylines (`bev_smooth`); still scored KITTI at 4 fps. |
| 3 | Detection gates, visual ego motion, curved ego path. Line tracker not yet `lines_mt`. |
| 4 | `capture_hz` time base, BEV-phase ego speed, `lines_mt` multi-line tracker, YOLOPv2 `decode_fix`. |
| 5 | Looser BEV-phase acceptance, clutter filters, first gore rules. Five clips crashed on `np.trapezoid` (needs NumPy 2); iter5b switched to `np.trapz` / a fallback and finished the set. Lane coverage dropped on lane split, Munich, and highway final. Exit gore never fired. |
| 6 | Parked cars pinned as dots; short lanes kept unless both short and young; gores must persist. New widen-gore rule: exit approach draws a merge area at frames 21–26. False gores on city, lane split, and A2D2 cleared; Munich gores 368 → 6. Diagonal / crossing edge filter (`FBEV_EDGE_MAX_DEG=15`) was too strict: lane-split left edge 93% → 40%, Munich right edge 92% → 64%. |
| 7 | Edge gate 25° with trim (instead of drop) plus a 5-frame ego-motion hold: edge presence back near iter5 (lane split L 40 → 93%, A2D2 76/72 → 97/94%), and no edge crosses a lane. Lane lines hugging a road edge (< 0.8 m) are suppressed (exit approach f20). Young and sideways-moving tracks are drawn as dots (lane-split f15 diagonal trail). Robust Kalman update (NIS > 9): Munich accel 0.912 → 0.676, A2D2a 0.901 → 0.775. Gores still only at exit approach f21–26 and Munich f760–765. Open: Munich R edge 72.7% (iter5 92.4), lane split accel 1.048, A2D2b accel 0.879. |

## Layout

```
foundation_bev/
  src/run_iter7.py     # GPU entry point
  src/                 # WildDet3D infer, BEV, lines_mt, ego motion, metrics
  tests/               # pytest, including a NumPy 1.x shim
  cams_iter7.json      # per-clip intrinsics and notes used by the runner
  SPEC.md              # iter7 change list and GPU-run contract
  patch.diff           # iter7 vs iter6 src
  tools/               # offline gore and edge-replay checks
  yolo_baseline/       # optional YOLOPv2 comparison only
  history/iterN_spec/  # earlier code and markdown
  results/iterN/       # metrics tables and Drive links
```
