# Foundation-model BEV pipeline

This directory is the foundation-model bird's-eye-view tracker. It is a separate line of work from the YOLO / TwinLiteNet pipeline on `lane-solver-fast` and `main`. The current code is **iter15** (`SPEC.md`, open issues in `results/iter15/KNOWN_ISSUES.md`), GPU-validated on a T4 (`results/iter15/`). iter15 fixes the ego motion: a yaw sign bug that mirrored every drawn turn since iter3, BEV-ECC yaw on the road mask, and a lane-parallelism pitch correction for the BEV-phase speed. It is validated against KITTI OXTS ground truth on the 6 KITTI clips. iter14 had no GPU run (public output identical to iter13). Earlier iteration specs are under `history/`, and small result tables are under `results/`.

Videos are not in this repo. They live in the private Google Drive folder [KITTI_BEV_Test_Videos](https://drive.google.com/drive/folders/1WlUGEUkBr-VqJg3jSeieE6nXRYUsdCq1). Per-iteration file IDs are in `results/iterN/DRIVE_LINKS.txt`.

## Pipeline

The run is two stages, then a BEV render. TwinLiteNet and YOLO are never used inside the foundation pipeline. YOLOPv2 is an optional measurement-only baseline (`yolo_baseline/`).

1. **WildDet3D** (stage 1) — open-vocabulary 3D boxes for `car`, `person`, `bicycle`, plus lane-line and road-edge prompt boxes.
2. **SAM 2.1** — box-prompted masks (lower-half fallback if a box is missing).
3. **Depth Anything V2** — per-pixel depth, lifted onto a ground-plane BEV.
4. **Tracking and lines**
   - ByteTrack-style association with a Kalman filter.
   - Ego-relative trails, capped at 2 s / 15 m. Cars are drawn red, people and bicycles in their own colours.
   - Near-stationary cars are pinned as a single dot with no trail; young tracks (< 5 rows), tracks moving more sideways than forward, and tracks first seen right of the ego that move backward (right-hand traffic: a parked car, not oncoming) are drawn as a dot at their current position (drawing only; tracks and metrics are unchanged).
   - `lines_mt` tracks lane lines and road edges separately, with split/merge links, fork events, and gore (merge-area) polygons. Next to a parked-car row the road edge is moved out to the kerb behind the cars (blended in and out over 0.3 s). Short holes (≤ 5 m) in a road edge are bridged before it is trimmed. Lanes and edges get a zero-lag centred temporal smoothing (±0.3 s) before rendering, the ego lane pair's width is smoothed jointly over ±1.5 s, and the ego pair's width offset is Kalman (RTS) smoothed. Lanes and main road edges confirmed in the first 0.6 s are backfilled into the first frames. Short diagonal edge fragments are vetoed before the hold, and a kerb blend change restarts the widen-gore history.
5. **Ego speed and yaw** — speed selection order: OXTS, then BEV phase correlation, then Lucas–Kanade flow, then a scene closing-rate, then the per-clip prior (iter14: road-masked BEV phase before the prior). iter15: the BEV-phase estimators are re-run at a ground pitch measured from lane parallelism (≥ 5 lane pairs); yaw comes from ECC alignment of consecutive road-masked BEV frames (+ = right turn, the convention of every consumer). Metrics use each clip's capture rate (`capture_hz`): KITTI 10 Hz, Munich 30 Hz, A2D2 10 Hz.

The BEV overlay shows only class-coloured trajectories, lane-line curves, road-edge curves, the ego path, and gore polygons. It does not show depth maps or edge maps.

## How to run iter15

Entry point: `src/run_iter15.py`. The dataset directory (`FBEV_DS`) must contain the clip MP4s, `cams.json` (intrinsics used at capture time), `motion_assumptions.json`, and this package (`src/`, `cams_iter13.json`, and optionally `yolo_baseline/`). The nine clips are `kitti_highway_sample`, `kitti_highway_exit_approach`, `kitti_highway_late`, `kitti_highway_final`, `kitti_city_cyclist_candidate`, `kitti_lane_split_candidate`, `munich`, `a2d2a`, and `a2d2b`.

Typical GPU run (one T4, about 75 minutes for stage 1 + stage 2 + the YOLO baseline):

```bash
export FBEV_DS=/path/to/dataset
export FBEV_OUT=/path/to/output
export FBEV_TMP=/path/to/scratch
# Optional YOLOPv2 baseline (measurement only):
# export FBEV_YOLOPV2=/path/to/yolopv2.torchscript
# export FBEV_YOLO_REPO=/path/to/sam-bev-22nov
python src/run_iter15.py
```

Outputs land in `FBEV_OUT`: `results/<clip>_foundation_bev_iter15.mp4`, `stills/`, `tracks/`, `polylines/` (with `step`, `edges_raw` and `edges_raw_held` for exact replays), `dets_dump/`, `metrics.md`, `ego_speed_table.md`, `metrics_yolo_vs_run.md`, and `gate.jsonl` (night-gate statistics; `tools/gate_summary.py`).

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
| `FBEV_EDGE_MAX_DEG` | `30` | Near the ego (z ≤ 20 m), trim a main road edge whose heading *converges* toward the road by more than this many degrees; a lane crossing invalidates everything beyond it, and the nearest good run (≥ 4 m) is kept |
| `FBEV_EDGE_DIVERGE_DEG` | `45` | Same, for heading that *diverges* away from the road (kerb flaring out at corners) |
| `FBEV_EDGE_EGO_HALF_M` | `1.2` | Edge samples closer than this to the ego centreline are dropped |
| `FBEV_EDGE_HOLD_S` | `0.5` | Edge hold time in seconds (5 frames at 10 Hz, 15 at 30 Hz); a held fill may only add samples |
| `FBEV_EDGE_BACKFILL_MODE` | `hold` | iter12: clip-start edge backfill holds the first live offset (`cv`: iter11 constant velocity) |
| `FBEV_EDGE_BACKFILL_COVER` | `1` | iter12: backfill source must reach z = 10 m; near fill of partial edges; diagonal copies rejected |
| `FBEV_EDGE_ANCHOR_S` / `FBEV_EDGE_ANCHOR_DEV` | `3.0` / `0.15` | iter12: a missing or hold-coasted main edge is drawn as nearest lane (else opposite edge) + learned offset, for up to 3 s |
| `FBEV_KERB_TRAFFIC` | `outer` | Kerb-rule traffic guard (`under` was tried in iter12 and rejected) |
| `FBEV_LANE_EDGE_DUP_M` | `0.8` | Drop a lane line (not the nearest one per side) that runs within this distance of a main edge over ≥ 70 % of their common span |
| `FBEV_FAR_MODE` | `ray` | Final smoothing pass only: range noise of far (z > 12 m, full by 18 m), off-axis (> 15°, full by 30°) cars lies along the viewing ray. `off` / `inflate` / `both` are experimental |
| `FBEV_RAY_R` | unset (off) | Experimental ray-aligned measurement noise in the online filter too; not used in GPU runs |
| `FBEV_GORE_GROW_M` | `1.2` | Widen-gore: gap must rise this many metres above the recent minimum |
| `FBEV_GORE_MAX_STEP` | `0.4` | Widen-gore: max gap growth per step (a jump restarts the history) |
| `FBEV_KERB_CARS` | `1` | Move a main edge that runs along parked-car bottoms out to the kerb line behind the cars (outward only, capped by `FBEV_KERB_MAX_PUSH_M=2.5`; cars with a long lane line alongside are treated as traffic and skipped; enter after `FBEV_KERB_ENTER=3` frames, hold `FBEV_KERB_HOLD=5`) |
| `FBEV_RHT_PIN` | `1` | Right-hand traffic: a track first seen right of the ego that moves backward (≤ `FBEV_RHT_MAX_RATIO=1.5` × ego speed) is drawn as a dot |
| `FBEV_LINE_SMOOTH_S` | `0.3` | Half-window (s) of the zero-lag centred lane / edge smoothing; symmetric window (`FBEV_LINE_SMOOTH_SYM=1`), frames moved by ego travel (`FBEV_LINE_SMOOTH_POSE=0`) |
| `FBEV_WORLD_PIN` | `0` | Experimental VO-heading world pin for parked cars; off (the VO heading does not fit track motion) |
| `FBEV_EDGE_GAP_M` | `5.0` | Bridge interior holes up to this long in a main road edge before trimming (only when the bridge runs within `FBEV_EDGE_GAP_SLOPE=0.5` of the road heading); the trim rules still judge the bridged samples |
| `FBEV_PAIR_W_MODE` | `kf` | Ego lane-pair width smoothing: per (left id, right id) run of ≥ `FBEV_PAIR_KF_MIN_S=1.0` s, the width offset at z 10 is RTS-smoothed as a random walk (`FBEV_PAIR_KF_MODEL=rw`, `FBEV_PAIR_KF_Q=0.1` m/√s, `FBEV_PAIR_KF_R=0.15` m); shorter runs use the iter10 `tri` window |
| `FBEV_PAIR_W_S` | `1.5` | Half-window (s) of the `tri` width smoothing; the wobbling line absorbs the change, changes above `FBEV_PAIR_W_MAXDEV/2` (0.25 m) fade out |
| `FBEV_LANE_BACKFILL_S` | `0.6` | Lanes confirmed in the first this-many seconds are copied back into the earlier frames (moved by ego travel; `backfill: true`) |
| `FBEV_EDGE_BACKFILL_S` | lane window | Same for main road edges, by role (`main_L` / `main_R`); the offset follows the edge's own early motion (`FBEV_EDGE_BACKFILL_CV=1`, clipped to ±0.5 m) |
| `FBEV_EDGE_FRAG_M` | `6.0` | Veto a new main-edge piece shorter than this that runs off the road heading by more than `FBEV_EDGE_FRAG_DEV=0.25` (not held, not kerb); the hold coasts instead. On-heading short far pieces are extended to z 10 (`FBEV_EDGE_FRAG_EXT_M=12`) |
| `FBEV_GORE_KERB_RESET` | `1` | A change of an edge's kerb blend weight restarts the widen-gore gap history |
| `FBEV_KERB_RAMP_S` | `0.3` | Kerb push blends in / out over this time (`0` = iter9 on/off switch) |
| `FBEV_EDGE_CAR_CLEAR` | `1` | iter13: synthetic (anchored / backfilled / held) edge points inside a car footprint (+`FBEV_EDGE_CAR_TOL=0.2` m) are shifted out to the kerb side as a whole line (`FBEV_EDGE_CAR_PUSH=line`, ≤ `FBEV_EDGE_CAR_W=2.0` m), else cut 0.5 m before the car (`FBEV_EDGE_CAR_TRIM=1`, ≥ `FBEV_EDGE_CAR_TRIM_MIN=4` m kept), else dropped; tracker coasts not pushed (`FBEV_EDGE_CAR_COAST=0`) |
| `FBEV_EDGE_ANCHOR_RATE` | `1.0` | iter13: LaneAnchor offset may change at most this many m/s |
| `FBEV_EDGE_FRAG_COAST` | `1` | iter13: the diagonal-fragment veto also applies to tracker coasts |
| `FBEV_EDGE_ANCHOR_BACKFILL` | `1` | iter13: frames before the first LaneAnchor fit get its offset (≤ 1 s) |
| `FBEV_LOWFPS_TIME` | `1` | iter13: below 10 fps every frame-count window keeps its 10 Hz time span (bit-identical on the 9 public clips) |
| `FBEV_DRAW_OCCLUDE` | `1` | iter13, display only: synthetic edges are not painted over car boxes in the camera view |
| `FBEV_EGO_ROAD` | `1` | iter14: road-masked BEV-phase ego tier, used only where iter13 falls back to the prior and only within the guard (valid >= 40 %, >= 5 pairs, no repeats, 0.33-3x prior) |
| `FBEV_TB_ADAPT` / `FBEV_ROAD_VPLANE` | `auto` | iter14: noise-adaptive thin-bright threshold / road vertical-plane gate, only under the sodium-night gate (`FBEV_NIGHT_S=80`, `FBEV_NIGHT_TOPV=25`, `FBEV_NIGHT_S2=105`, `FBEV_NIGHT_TOPV2=90`); `0` off, `1` always |
| `FBEV_GATE_LOG` | `$FBEV_OUT/gate.jsonl` | iter14 diagnostics: per-call night-gate statistics with the clip tag |
| `FBEV_EDGE_CAR_BF` | `push` | iter14 knob: `trim` cuts clip-start backfilled edges before a parked car (city wjit 0.054 -> 0.037, z20 presence cost) |
| `FBEV_GORE_SPATIAL` | `0` | iter14 experiment: one-frame lane-edge divergence wedge, 0.5 s confirmation (false gores on public clips) |
| `FBEV_EGO_YAW` | `bev` | iter15: ego yaw source; `bev` = ECC on the road-masked BEV, `em` = point-LK EgoMotion with the sign fix, `legacy` = iter3-14 (mirrored turns) |
| `FBEV_EGO_PITCH` | `1` | iter15: lane-parallelism pitch correction of the BEV-phase speed (`0` = iter14) |
| `FBEV_TOPO_LOG` | unset | iter15 diagnostics only: per-frame road-topology log |
| `FBEV_TRIM_DEBUG` | unset | Diagnosis only: JSONL file of every edge trim's masks and kept run |

CPU tests (no GPU, no weights):

```bash
pytest -q tests
PYTHONPATH=tests/np1_sim pytest -q tests   # NumPy 1.x compatibility (`np.trapz` fallback)
```

## Changelog

Accel is RMS BEV acceleration of drawn tracks, in m/s². Lane figures are left/right presence at z = 10 m, in percent of frames. Iter2 and iter3 scored KITTI at the stored 4 fps, which understated acceleration; iter4 onward uses `capture_hz` (KITTI 10 Hz). Munich (30 Hz) and A2D2 (10 Hz) are consistent across iterations. Iter5 is the merged 9-clip result (4 highway clips from the first iter5 run + 5 clips recovered in iter5b after a NumPy 1.x `np.trapezoid` crash). Full tables are in `results/iterN/`.

### Accel RMS (m/s²)

| clip | iter2 | iter3 | iter4 | iter5 | iter6 | iter7 | iter8 | iter9 | iter10 | iter11 | iter12 | iter13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| highway sample | 0.504† | 0.584† | 0.108 | 0.095 | 0.095 | 0.095 | 0.095 | 0.095 | 0.095 | 0.095 | 0.095 | 0.095 |
| exit approach | 0.628† | 0.679† | 0.541 | 0.546 | 0.546 | 0.546 | 0.546 | 0.546 | 0.546 | 0.546 | 0.546 | 0.546 |
| highway late | 1.697† | 1.827† | 0.651 | 0.575 | 0.575 | **0.526** | 0.534 | 0.534 | 0.534 | 0.534 | 0.534 | 0.534 |
| highway final | 0.838† | 0.915† | 0.626 | 0.626 | 0.638 | 0.614 | **0.574** | 0.574 | 0.574 | 0.574 | 0.574 | 0.574 |
| city cyclist | 0.474† | 0.585† | 0.566 | 0.566 | 0.566 | 0.566 | **0.548** | 0.548 | 0.548 | 0.548 | 0.548 | 0.548 |
| lane split | 1.131† | 1.134† | 1.052 | 1.074 | **0.997** | 1.048 | **0.951** | 0.951 | 0.951 | 0.951 | 0.951 | 0.951 |
| munich | 1.079 | 0.911 | 0.917 | 0.911 | 0.912 | **0.676** | **0.655** | 0.655 | 0.655 | 0.655 | 0.655 | 0.655 |
| a2d2a | 1.261 | 0.820 | 0.774 | 0.791 | 0.901 | **0.775** | **0.495** | 0.495 | 0.495 | 0.495 | 0.495 | 0.499 |
| a2d2b | 1.119 | 0.890 | 0.890 | 0.890 | **0.827** | 0.879 | **0.821** | 0.821 | 0.821 | 0.821 | 0.821 | 0.821 |

† KITTI scored at 4 fps (not comparable to iter4+).

### Lane presence L10 / R10 (%)

| clip | iter4 | iter5 | iter6 | iter7 | iter8 | iter9 | iter10 | iter11 | iter12 | iter13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| highway sample | 95 / 95 | 92.5 / 92.5 | 95 / 95 | 95 / 95 | 95 / 95 | 95 / 95 | **100 / 100** | 100 / 100 | 100 / 100 | 100 / 100 |
| exit approach | 95 / 90 | 92.5 / 92.5 | 95 / 95 | 95 / 95 | 95 / 95 | 95 / 95 | **100 / 100** | 100 / 100 | 100 / 100 | 100 / 100 |
| highway late | 92.5 / 95 | 87.5 / 92.5 | 92.5 / 95 | 92.5 / 95 | 92.5 / 95 | 92.5 / 95 | **97.5 / 100** | 97.5 / 100 | 97.5 / 100 | 97.5 / 100 |
| highway final | 93.3 / 90 | 83.3 / 86.7 | 93.3 / 90 | 93.3 / 90 | 93.3 / 90 | 93.3 / 90 | 93.3 / 90 | **100 / 100** | 100 / 100 | 100 / 100 |
| city cyclist | 93.3 / 93.3 | 90 / 90 | 93.3 / 93.3 | 93.3 / 93.3 | 93.3 / 93.3 | 93.3 / 93.3 | **100 / 100** | 100 / 100 | 100 / 100 | 100 / 100 |
| lane split | 93.3 / 76.7 | 90 / 46.7 | 86.7 / 66.7 | **93.3 / 73.3** | 93.3 / 73.3 | 93.3 / 73.3 | **100** / 73.3 | 100 / 73.3 | 100 / 73.3 | 100 / 73.3 |
| munich | 87.6 / 65.4 | 84.9 / 56.9 | 85.7 / 65.3 | 85.7 / 65.3 | 85.7 / 65.3 | 85.7 / 65.2 | 85.9 / 65.4 | 85.9 / 65.4 | 85.9 / 65.4 | 85.9 / 65.4 |

### Road-edge presence L10 / R10 (%)

| clip | iter5 | iter6 | iter7 | iter8 | iter9 | iter10 | iter11 | iter12 | iter13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| highway late | 95 / 87.5 | 95 / 77.5 | 95 / 87.5 | 95 / 87.5 | 95 / 87.5 | 95 / 87.5 | **100 / 100** | 100 / 100 | 100 / 100 |
| lane split | 93.3 / 93.3 | 40 / 83.3 | 93.3 / 83.3 | **93.3 / 93.3** | 93.3 / 93.3 | 93.3 / 93.3 | **100 / 100** | 100 / 100 | 100 / 100 |
| munich | 99.2 / 92.4 | 87.9 / 64.0 | 93.9 / 72.7 | **99.8 / 84.3** | 99.8 / 84.1 | 99.8 / **86.2** | **100** / 86.3 | 100 / **95.8** | 100 / 96.8 |
| a2d2a | 79.7 / 77.3 | 76.2 / 71.5 | 96.5 / 94.2 | **97.1 / 94.8** | 95.3 / 94.8 | 95.9 / 94.8 | 97.7 / 94.8 | 97.7 / **98.8** | 97.7 / 97.7 |
| a2d2b | 97.1 / 97.1 | 85.5 / 83.7 | 97.1 / 95.9 | **97.1 / 97.1** | 97.1 / 97.1 | 97.1 / 97.1 | 97.1 / **100** | 97.1 / 100 | 99.4 / 100 |

Other clips are unchanged across iter5–iter10 (sample, exit approach 95 / 95; final 90 / 86.7; city 93.3 / 93.3). iter11 backfills edges at the clip start: sample, exit approach, final and city are 100 / 100.

### Line jitter at z = 10 m (iter8 → iter9, m)

iter9 smooths lanes and edges with a zero-lag centred window. Lane jit2 L / R, lane width jitter, edge jit2 L / R:

| clip | lane jit2 | lane wjit | edge jit2 |
|---|---|---|---|
| exit approach | 0.007/0.018 → 0.005/0.018 | 0.033 → 0.023 | 0.019/0.033 → 0.016/0.024 |
| highway final | 0.039/0.006 → 0.012/0.004 | 0.024 → 0.013 | 0.033/0.040 → 0.029/0.025 |
| city cyclist | 0.004/0.010 → 0.003/0.006 | 0.008 → 0.007 | 0.065/0.026 → 0.018/0.010 |
| lane split | 0.063/0.025 → 0.011/0.022 | 0.136 → 0.131 | 0.069/0.077 → 0.042/0.042 |
| munich | 0.009/0.003 → 0.003/0.002 | 0.010 → 0.006 | 0.034/0.042 → 0.010/0.024 |

### Lane width jitter and YOLOPv2 (iter9 → iter10, m)

iter10 smooths the ego lane pair's width jointly. Lane wjit at z = 10 m (YOLOPv2 measurement baseline in brackets):

| clip | lane wjit | lane jit2 R |
|---|---|---|
| exit approach | 0.023 → **0.017** (0.007) | 0.018 → 0.014 (0.009) |
| highway sample | 0.006 → 0.004 (0.006) | 0.003 → 0.003 (0.005) |
| highway late | 0.020 → 0.019 (0.009) | 0.007 → 0.007 (0.014) |
| highway final | 0.013 → 0.012 (0.008) | 0.004 → 0.004 (0.017) |
| city cyclist | 0.007 → **0.004** (0.005) | 0.006 → 0.003 (0.004) |

### Lane width jitter and YOLOPv2 (iter10 → iter11, m)

iter11 Kalman-smooths the ego lane pair's width offset. Lane wjit at z = 10 m (YOLOPv2 in brackets); edge wjit is the road-edge pair:

| clip | lane wjit | edge wjit |
|---|---|---|
| exit approach | 0.017 → **0.007** (0.007) | 0.048 → 0.047 |
| highway sample | 0.004 → 0.001 (0.006) | 0.019 → 0.022 |
| highway late | 0.019 → **0.006** (0.009) | 0.035 → 0.041 |
| highway final | 0.012 → **0.006** (0.008) | 0.023 → 0.026 |
| city cyclist | 0.004 → 0.001 (0.005) | 0.054 → 0.076 |
| lane split | 0.131 → 0.131 (0.092) | 0.127 → 0.115 |

### Edge width jitter (iter11 → iter12, m)

iter12 holds the clip-start edge backfill at the first live offset instead of extrapolating its drift. Edge wjit at z = 10 m
(live-only, i.e. without backfilled or anchored frames, in brackets: unchanged):

| clip | edge wjit |
|---|---|
| highway sample | 0.022 → 0.019 |
| exit approach | 0.047 → **0.042** (0.048) |
| highway late | 0.041 → **0.034** (0.035) |
| highway final | 0.026 → 0.022 (0.023) |
| city cyclist | 0.076 → **0.037** (0.054) |
| munich | 0.041 → 0.041; R10 presence 86.3 → **95.8** |
| a2d2a | 0.065 → 0.062; R10 presence 94.8 → **98.8** |

Lane wjit is unchanged (lane split 0.131 vs YOLOPv2 0.092: structural).

### Edges and car footprints (iter12 → iter13)

iter13 keeps synthetic road edges out of parked-car footprints. Car audit (frames with a synthetic edge inside a car
footprint): 16 → 0. Munich R z10 862 → 871 / 900, edge presence R 95.8 → 96.8 %, edge wjit 0.041 → 0.037; a2d2b L edge
97.1 → 99.4 %. Gores identical (exit f21–26, Munich f760–765), idsw 0, no lane presence change. Disclosed regression:
city edge wjit 0.037 → 0.054 (`FBEV_EDGE_CAR_CLEAR=0` restores iter12).

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
| 8 | Direction-aware edge trim (converging > 30°, diverging > 45°), a lane crossing invalidates the rest of the edge, ego-corridor guard, nearest run kept, 0.5 s time-based hold that may only add samples; flared edges seed no fork gore. Munich R edge 72.7 → 84.3%, lane split R 83.3 → 93.3%, A2D2b R 95.9 → 97.1%; no edge crosses a lane or enters the ego corridor. Final smoothing pass treats far, off-axis range noise as ray-aligned: A2D2a accel 0.775 → 0.495 (far car 1.84 → 0.43), lane split 1.048 → 0.951, A2D2b 0.879 → 0.821. Gores still only at exit approach f21–26 and Munich f760–765; idsw 0. |
| 9 | Road edge moved out to the kerb behind parked-car rows (lane split f15–29, Munich, A2D2a), with a traffic-queue guard and enter/hold hysteresis. Parked cars on the right whose track drifts backward are drawn as dots (lane split f29, Munich f692); the VO-heading world pin was tried and rejected. Zero-lag centred lane / edge smoothing (±0.3 s): lane split lane L jit2 0.063 → 0.011, final 0.039 → 0.012, Munich edges 0.034/0.042 → 0.010/0.024. Polylines now carry untrimmed main edges and the ego pose. Accel, gores, held frames and idsw unchanged; lane split edge width jitter 0.107 → 0.125. |
| 10 | Road-edge holes ≤ 5 m bridged before the trim (Munich R edge 84.1 → 86.2%; holes behind parked cars made the trim skip to the far part); flares stay trimmed. Ego lane-pair width smoothed jointly over ±1.5 s (exit wjit 0.023 → 0.017, city 0.007 → 0.004). Lanes backfilled into the first 0.3 s (sample / exit / city 100 / 100, late 97.5 / 100, lane split L 100). Kerb push blended in and out over 0.3 s (A2D2a edge wjit 0.070 → 0.065, Munich 0.043 → 0.040). Accel, idsw (0) and held frames unchanged. **Known issue: a false widen gore on Munich f881–885 (fixed in iter11);** exit f21–26 and Munich f760–765 still fire. Munich R also lost f520–526 / f557–560 at z = 10. |
| 12 | Clip-start edge backfill holds the first live offset (city edge wjit 0.076 → 0.037, late 0.041 → 0.034) and must cover z = 10 m with no diagonal copy; edge window 1.0 s (A2D2a f0 right edge). LaneAnchor: a missing or hold-coasted main edge follows the nearest lane (else the opposite edge) at its learned offset for up to 3 s (Munich R z10 777 → 862/900, f165–166 present). Polylines carry the tracker coast flag for exact CPU replays. Gores, accel, tracks and idsw (0) unchanged. **Known issues:** the anchored Munich right edge is drawn over parked cars (f189–251), and on some frames jumps inside them; Munich f562–563 / f570–571 still absent. |
| 11 | Munich f881–885 false widen gore removed: the kerb ramp widened the lane–edge gap gradually, so a kerb blend change now restarts the widen history (gores only exit f21–26 and Munich f760–765). Short diagonal edge fragments are vetoed before the hold, so the hold coasts (Munich R z10 776 → 777: +f520–526, f550–560, f615; −f561–577, f618). Main road edges backfilled at the clip start, and the window grew 0.3 → 0.6 s (edges 100 % on every KITTI clip; final lanes 93.3 / 90 → 100 / 100). Ego lane-pair width offset RTS-smoothed as a random walk: lane wjit exit 0.017 → 0.007, late 0.019 → 0.006, final 0.012 → 0.006, all at or below YOLOPv2. Accel, tracks and idsw (0) unchanged. Edge wjit rose on city (0.054 → 0.076) and late (0.035 → 0.041). |
| 13 | Synthetic edges (anchored / backfilled / held) cleared out of parked-car footprints by a whole-line shift, else trimmed (car audit 16 → 0 frames; Munich R z10 862 → 871, a2d2b f0 L edge). LaneAnchor rate limit 1 m/s; fragment veto on tracker coasts; anchor backfill; low-fps time base (fps < 10 keeps 10 Hz time spans; public clips bit-identical); front-view occlusion of synthetic edges. City edge wjit 0.037 → 0.054 (disclosed). |
| 15 | Ego motion. Yaw sign fixed (drawn turns were mirrored since iter3; lane split yaw MAE vs OXTS 24.4 → 1.1 deg/s) and yaw from BEV-ECC on the road mask: yaw MAE lower on all 6 KITTI clips (0.11–1.13 deg/s). Lane-parallelism pitch for the BEV-phase speed: KITTI mean |speed error| vs OXTS 7.7 → 5.6 % (late −9.2 → +0.4 %, exit −4.0 → +2.6 %; final worse, −1.5 → +4.0 %), path end error lower on 5 / 6. Tracks, accel, gores and presence unchanged; lines move only where the speed changed (late edge wjit 0.034 → 0.041, within noise). |
| 14 | No GPU run (public output identical to iter13, CPU-verified). For private night / 4 fps footage: road-masked BEV-phase ego tier (guarded), noise-adaptive thin-bright threshold and road vertical-plane gate behind a sodium-night gate (0 firings on the 9 public clips). Night split: false gores 21 → 0, ego lane z10 0.93 / 0.72 → 0.22 / 0.05 m, left edge 3.95 → 1.12 m, ego distance −48 % → +16 %; day merge ego distance +75 % → +3 %. Fork / merge gores still open. |

## Layout

```
foundation_bev/
  src/run_iter15.py    # GPU entry point
  src/                 # WildDet3D infer, BEV, lines_mt, ego motion, metrics
  tests/               # pytest, including a NumPy 1.x shim
  cams_iter13.json     # per-clip intrinsics and notes used by the runner (unchanged since iter12)
  SPEC.md              # iter15 change list and verification
  patch.diff           # iter15 vs iter14 src
  tools/               # offline gore, edge, track, smoothing / kerb, dot-rule, lane-pair and gap replays
  tools/ego_eval.py, tools/kitti_gt/  # iter15 ego error vs KITTI OXTS (raw-drive matching by range reads)
  kitti_gt_map.json    # KITTI clip -> raw drive / start frame
  yolo_baseline/       # optional YOLOPv2 comparison only
  history/iterN_spec/  # earlier code and markdown
  results/iterN/       # metrics tables and Drive links
```
