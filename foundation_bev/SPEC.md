# iter8 SPEC: FINAL (24 Nov 2026, ~11:10 IST)
iter8 is iter7 plus the edge and far-track fixes below. Render rules, online association, relink and the gore rules are unchanged. NumPy-1 compat (`_trapz`) and `FBEV_REUSE_DETS` still work. There is no TwinLiteNet or YOLO in the pipeline; YOLOPv2 is only the measurement baseline.
`patch.diff` is a diff against `iter7_spec/src`. It covers `lines_mt.py`, `bev_smooth.py` and `foundation_bev.py`, and renames `run_iter7.py` to `run_iter8.py`.

## Diagnosis (iter7 GPU outputs)
- **munich R 72.7 / L 93.9** (iter5: 92.4 / 99.2).
  - On the iter5 raw polylines, the 25° gate trimmed 199 munich-R frames. All of them were *diverging* (the kerb flares away from the road at corners and driveways, 30–55°), and none crossed a lane. The converging trims (munich L) did all cross lanes.
  - The iter7 hold was 5 *frames*, which is 0.17 s at munich's 30 fps.
  - The rule kept the *longest* run, which is often the far part beyond 10 m.
- **lane_split R 83.3**, f23–29: iter7 kept the far diagonal run through the ego lane and dropped the good near part.
- **a2d2b R 95.9**: same mechanism, a far run kept after a crossing.
- **Accel regressions.** The cause is stage-1 run-to-run variance plus ego yaw, not the iter7 code: iter6 and iter7 code on the same iter7 dets give lane_split 0.896 vs 0.897.
  - lane_split 1.048: tracks **6** and **7**, parked cars on the right (x ≈ 4.1–4.5, z ≈ 12–15, f0–29), per-track RMS 1.75 and 1.44.
  - a2d2b 0.879: track **8** (x −4.3, z 11.8, f63–100), RMS 1.58.
  - a2d2a: far car track **31** (the brief's track 30; x −12, z 19–25, f107–170). It has 4–5 m range jumps along the viewing ray, RMS 1.84.

## Changes
1. **Direction-aware edge trim** (`lines_mt.trim_edge`).
   - Near the ego (z ≤ 20 m), heading that *converges* toward the road by more than 30° is bad (`FBEV_EDGE_MAX_DEG=30`).
   - Heading that *diverges* away from the road is bad only beyond 45° (`FBEV_EDGE_DIVERGE_DEG=45`). 60° gives munich R 94.1 on the replay but keeps 45–60° corner flares.
   - A lane crossing (≥ 2 samples more than 0.5 m inside a lane that starts inside the edge) invalidates everything from the first crossing sample onwards.
   - **Ego-corridor guard:** samples with |x| < 1.2 m on the edge's own side are bad (`FBEV_EDGE_EGO_HALF_M=1.2`).
   - The **nearest** good run of at least 4 m is kept, instead of the longest.
2. **Time-based, monotone hold** (`EdgeHold`, `retrim_held`).
   - The hold is 0.5 s (`FBEV_EDGE_HOLD_S`): 5 frames at 10 fps and 15 at munich's 30 fps. The decay is rescaled per frame.
   - A fill may only *add* samples. If re-trimming the filled edge would lose a measured sample, the current edge is used unchanged (city f23).
3. **Flare flag** (`is_flared`, `fork_gores`). A main edge that keeps a diverging deviation above 30° is flagged `flare`. The split_in-vs-main fork gore is skipped for flared edges. This removes a new munich f587–593 / f580–585 gore that the wider gate would otherwise create.
4. **Far-track measurement model, final smoothing pass only** (`bev_smooth._KF(final=True)`, `ObjTracker._smooth_final`).
   - Monocular range noise lies along the viewing ray. For cars that are far (z > 12 m, full by 18 m) *and* off-axis (bearing > 15°, full by 30°), R is ray-aligned (`FBEV_FAR_MODE=ray`, `FBEV_FAR_Z0=12`, `FBEV_FAR_RAMP=6`, `FBEV_RAY_DEG0=15`).
   - Near and on-axis tracks are unchanged.
   - The online filter, association, relink and the stationary decision keep the iter7 model, so IDs, fragments and relinks do not change.
   - Rejected after replay:
     - **Range inflation** (`FBEV_FAR_MODE=inflate|both`): lower accel, but smoothed tracks drift 5–10 m from the measurements on the KITTI highways (it stops following the data). It also caused a CPU-smoke idsw on late.
     - **Tighter final NIS gate** (4 or 6): larger residuals.
     - **True-world re-pin** (`FBEV_REPIN`): no effect.
     All are off by default.

## Offline edge replay (`tools/replay_edges.py`, edge L/R presence % at z = 10)
**On the iter5 raw GPU polylines** (the edges before any gate): `replay/edge_replay_iter5_polylines.md`

| clip | iter5 raw | iter7 rule | iter7 GPU | **iter8 rule** |
|---|---|---|---|---|
| munich | 99.2/92.4 | 94.1/75.9 | 93.9/72.7 | **99.7/88.2** |
| lane_split | 93.3/93.3 | 93.3/86.7 | 93.3/83.3 | **93.3/93.3** |
| a2d2b | 97.1/97.1 | 97.1/95.3 | 97.1/95.9 | **97.1/97.1** |
| a2d2a | 79.7/77.3 | 96.5/94.8 | 96.5/94.2 | **97.1/94.8** |
| city, exit, final, late, sample | | same as iter7 | | 93.3/93.3, 95/95, 90/86.7, 95/87.5, 95/95 |

**On the iter7 GPU polylines** (already trimmed, so only the hold and retrim can act): `replay/edge_replay_iter7_gpu_polylines.md`
- munich 93.9/72.7 → **95.9/76.7**
- lane_split 93.3/83.3 → **93.3/86.7**
- a2d2b 97.1/95.9 → **97.1/97.1**
- a2d2a 96.5/94.2 → **96.5/94.8**
- all others unchanged

Other replay checks:
- Lane presence is unchanged on every clip. Edge jit2 is within ±0.01, except munich R and lane_split R (0.07 → 0.078).
- **Edge records crossing a lane: 0 everywhere.** Records entering the ego corridor (iter5 polylines → iter7 rule → iter8 rule):
  - lane_split 22 → 14 → 0
  - munich 47 → 30 → 0
  - a2d2a 36 → 36 → 0
  - a2d2b 30 → 36 → 0
- On the iter7 GPU polylines, munich 30, lane_split 13, a2d2a 36 and a2d2b 36 all go to 0.
- **Offline DivergenceGore** (`tools/check_divergence.py`): the same or fewer gores than the iter7 rule. On the iter5 polylines, munich's f592 widen is removed and exit f19–22 is unchanged. This offline check does not exactly reproduce the GPU gores (exit f21–26, munich f760–765), so please verify on GPU.

## Offline track replay (`tools/replay_tracks.py`, iter7 GPU tracks.csv, association fixed)
accel_rms m/s² (idsw proxy 0 and n_tracks unchanged on all clips): `replay/track_replay_accel.md`

| clip | iter7 GPU | iter7 model replay | **iter8 model** |
|---|---|---|---|
| a2d2a | 0.775 | 0.775 | **0.495** |
| a2d2b | 0.879 | 0.879 | **0.821** (iter6 0.827) |
| lane_split | 1.048 | 1.048 | **0.951** (iter6 0.997) |
| city | 0.566 | 0.566 | 0.548 |
| final | 0.614 | 0.614 | 0.574 |
| late | 0.526 | 0.526 | 0.534 |
| munich | 0.676 | 0.671 | 0.650 |
| exit, sample | 0.546, 0.095 | same | same |

- Culprit tracks, per-track RMS (`replay/accel_culprits_per_track.txt`):
  - a2d2a track 31: 1.84 → 0.43
  - a2d2a track 33: 1.01 → 0.56
  - lane_split track 6: 1.75 → 1.48
  - lane_split track 7: 1.44 → 1.24
  - a2d2b track 8: 1.58 → 1.47
- The median and p90 distance from the smoothed track to the measurements are unchanged (`replay/track_replay_accel_residual.md`), so the gain is not oversmoothing.

## Tests (CPU, 24 Nov ~11:05 IST)
- pytest: **22 passed** with NumPy 2, and **22 passed** under the NumPy-1 simulation (`PYTHONPATH=tests/np1_sim`). The 4 legacy scripts print ALL PASSED.
  - New tests:
    - crossing-invalidates-beyond, on the lane_split f23 geometry (the iter7 rule keeps the far part; iter8 keeps the near part);
    - the ego corridor;
    - converging 35° trimmed, diverging 40° kept and flagged, diverging 50° trimmed;
    - fps-based hold (10 fps → 5 frames, 30 fps → 15);
    - a fill can't cost measured samples;
    - flared main edge → no fork gore;
    - the far model applies only in the final pass and leaves near tracks identical.
- CPU smoke, all 9 clips, NumPy-1 simulation:
  - Inputs: iter7 GPU stage-1 dets plus classical masks, iter8 code vs iter7 code on the same inputs.
  - Result: rc 0 on all 9 clips. Full tables: `cpu_smoke/smoke_table.md` and `cpu_smoke/gores_held.md`.

| clip (CPU) | accel iter7 → iter8 | edge L/R pres | idsw |
|---|---|---|---|
| lane_split | 0.897 → 0.817 | 63.3/20.0 → 86.7/73.3 | 0 → 0 |
| exit_approach | 0.492 → 0.492 | 90.0/87.5 → 90.0/95.0 | 0 → 0 |
| munich | 0.663 → 0.649 | 93.9/65.2 → 96.3/83.7 | 0 → 0 |
| a2d2a | 0.937 → 0.811 | 75.6/79.7 → 80.2/88.4 | 0 → 0 |
| a2d2b | 0.657 → 0.622 | 87.8/90.1 → 89.5/97.1 | 0 → 0 |
| city / final / late / sample | 0.566→0.548 / 0.551→0.517 / 0.685→0.645 / 0.196→0.196 | same or higher | 0 → 0 |

  - Lane presence is identical on all clips.
  - CPU gores: munich f580–585 (iter7 code) → none; lane_split f3–7 appears with both codes and is a classical-mask artifact.
  - Held-edge frames: munich 175 → 184 and a2d2a 41 → 47; the others are equal.
  - Stills (munich 692/760, lane_split 20, a2d2a 119) checked by eye. lane_split's right edge no longer runs through the ego lane.
- Kernel smoke (`FBEV_SMOKE=1 FBEV_MAXF=8`, NumPy-1 simulation): rc 0, fails none, 9 × `*_foundation_bev_iter8.mp4`.

## GPU Bot run
- Entry: `src/run_iter8.py`. Dataset = this folder: `src/`, `cams_iter8.json` (identical to iter7), `yolo_baseline/`, plus the same clip mp4s, cams.json and motion_assumptions.json as iter7.
- **Free credit only.** Clips: all 9 (leave `FBEV_TAGS` unset).
- **Full stage 1+2** (no `FBEV_REUSE_DETS`).
- Env vars:
  - `FBEV_OUT`, `FBEV_TMP`, `FBEV_DS` as before.
  - `FBEV_EXTRA_STILLS=6,7,8,15,18,20,22,24,26,29,34,63,80,86,100,107,119,121,123,133,140,180,450,692,697,760,762`
    - This is the iter7 set plus munich 692 and the accel-culprit frames:
      - lane_split tracks 6/7: f0–29 (stills 6, 7, 8, 15, 20, 29);
      - a2d2b track 8: 63, 80, 86, 100;
      - a2d2a track 31: 107, 119, 121, 123, 140.
  - `FBEV_YOLOPV2` + `FBEV_YOLO_REPO`: YOLO baseline only, as before.
  - Do not set `FBEV_SMOKE`, `FBEV_MAXF`, `FBEV_RAY_R`, `FBEV_REPIN` or `FBEV_ROBUST_NIS`.
  - Defaults, so no need to set them: `FBEV_EDGE_MAX_DEG=30`, `FBEV_EDGE_DIVERGE_DEG=45`, `FBEV_EDGE_EGO_HALF_M=1.2`, `FBEV_EDGE_HOLD_S=0.5`, `FBEV_FAR_MODE=ray`, `FBEV_FAR_Z0=12`, `FBEV_FAR_RAMP=6`, `FBEV_RAY_DEG0=15`, `FBEV_LANE_EDGE_DUP_M=0.8`, `FBEV_GORE_GROW_M=1.2`, `FBEV_GORE_MAX_STEP=0.4`.
- Outputs:
  - `results/<clip>_foundation_bev_iter8.mp4`
  - `stills/`
  - `tracks/`
  - `polylines/` (edges carry the `held` and `flare` flags)
  - `dets_dump/`, `dets/`
  - `metrics.md`, `ego_speed_table.md`, `metrics_yolo_vs_run.md`
- Expected T4 time: **~75–80 min**, the same as iter7 (stage-2 changes are negligible).
- **Please produce `metrics_vs_iter7.md`**:
  - one 9-clip table, iter7 (`/workspace/foundation_bev/iter7`) → iter8;
  - columns: accel_rms, n_tracks, idsw, lanes L10/R10 pres + jit2, edges L10/R10 pres + jit2;
  - gore frames per clip: they must fire only on exit_approach f21–26 and munich f760–765;
  - held-edge frames per clip;
  - per-track accel RMS for the culprits: lane_split tracks near x ≈ 4.1–4.5 / z 12–15, f0–29; a2d2b x ≈ −4.3 / z 12, f63–100; a2d2a x ≈ −12 / z 21, f107–170. Track IDs may differ because stage 1 is not deterministic;
  - flag any clip whose lane or edge presence drops by more than 5 points vs iter7, or whose idsw is above 0. Check munich R, lane_split R, a2d2b R and a2d2a accel first.
- Caveat: stage 1 is not deterministic between runs, so accel can move by about ±0.05 independently of this code. Use the culprit per-track table to separate the two.
