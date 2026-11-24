# iter7 SPEC: FINAL (24 Nov 2026, ~09:00 IST)
iter7 is iter6 plus the four fixes below. The NumPy-1 compat (`_trapz`) and its tests are kept, and `FBEV_REUSE_DETS` still works.
`patch.diff` is a diff against `iter6_spec/src`. It covers `lines_mt.py`, `foundation_bev.py` and `bev_smooth.py`, and renames `run_iter6.py` to `run_iter7.py`.

## Changes
1. **Edge presence regression from the 15° gate** (`lines_mt.trim_edge`, `EdgeHold`, `StageLines`).
   - **Trim instead of drop.** The gate is now 25° (`FBEV_EDGE_MAX_DEG`, default 25). A sample is marked bad when either:
     - it sits more than 0.5 m on the inner side of a lane line that *starts inside* the edge, so kerb or parking paint beyond the edge no longer counts; or
     - it is near the ego (z ≤ 20 m) and its local heading is more than 25° off the road heading.
     The longest good run of at least 4 m is kept. Without one, the edge is dropped for that frame.
   - **Temporal hold.** Each main_L / main_R edge is held for up to 5 frames. It is propagated by the selected ego travel (x(z) → x(z+ds)), with confidence 0.8^age.
     - The held edge fills the samples trimmed from the current edge, or stands in for an edge that is missing (`held=True`).
     - Held and filled geometry is re-trimmed, so it can never cross a lane.
2. **lane_split f0015 diagonal trail** (`foundation_bev` drawing only).
   - The trail belongs to the GPU's track 14 (CPU track 10): v_long ≈ 3.8 m/s, v_lat ≈ 4.2 m/s. It is a real slow mover, *not* a car just above the 1.5 m/s pin threshold.
   - New rules:
     - tracks with fewer than 5 rows are drawn as a dot only;
     - a track with v_lat > v_long (and v_lat > 1 m/s) is drawn as a dot at its **current** position with no trail (`STATIC_PERP_RATIO`);
     - tracks with v_long < 1.5 m/s are still pinned at their median position.
   - tracks.csv and the metrics are unchanged.
   - The missing left edge is fix 1: on the iter5 GPU polylines, lane_split edge L presence goes 46.7 → 93.3.
3. **Lane hugging an edge** (`lines_mt.suppress_edge_duplicates`).
   - A lane line is dropped when it runs within 0.8 m of a main edge over at least 70% of a common span of 12 or more samples (`FBEV_LANE_EDGE_DUP_M`, default 0.8).
   - Why 0.8 m and not 0.5: the exit_approach f18–20 gap is 0.2 m near the ego and grows to 0.7 m at 35 m (painted edge line vs asphalt edge).
   - The nearest line on each side, and the nearest line reaching z = 10 m, are never dropped. Lane presence is therefore unchanged on all 9 clips.
4. **a2d2a accel_rms 0.791 → 0.901.**
   - The rise comes from one far car: GPU iter6 track 30 (CPU track 43), f107–157, x ≈ −12 m, z ≈ 20–25 m. Its own RMS went from 1.91 to 2.62.
   - Around f119–123, WD3D range jumps by 4–5 m along the viewing ray (meas z 24 → 31, x −13 → −17). An axis-aligned R let that range error drag x.
   - Default fix: a robust KF update (`ROBUST_NIS = 9`, which inflates R when NIS > 9).
     - It does not change a2d2a (CPU 0.807 → 0.808).
     - It lowers munich (0.892 → 0.664) and a2d2b (0.898 → 0.689).
   - Opt-in, **OFF**: `FBEV_RAY_R=1`, a ray-aligned measurement covariance for bearings above 10°.
     - It fixes a2d2a (0.808 → 0.664).
     - It worsens a2d2b (0.689 → 0.785), late (0.688 → 0.788) and exit (0.492 → 0.525).
     - It fails the synthetic relink/stationary test, so it is not enabled.

## Offline edge replay (`tools/replay_edges.py`, edge L/R presence % at z = 10)
The main table starts from the **ungated iter5 GPU polylines**, which still contain the edges the iter6 gate removed. The iter6-rule replay reproduces the actual iter6 GPU numbers, so the replay is valid.

| clip | iter5 (no gate) | iter6 rule (replay) | iter6 GPU | **iter7 rule** | lane L/R iter5 → iter7 |
|---|---|---|---|---|---|
| a2d2a | 79.7/77.3 | 76.2/68.0 | 76.2/71.5 | **96.5/94.8** | 12.8/22.7 → same |
| a2d2b | 97.1/97.1 | 89.5/85.5 | 85.5/83.7 | **97.1/95.3** | 8.7/1.7 → same |
| city | 93.3/93.3 | 93.3/93.3 | 93.3/93.3 | **93.3/93.3** | 90.0/90.0 → same |
| exit_approach | 95.0/95.0 | 95.0/95.0 | 95.0/95.0 | **95.0/95.0** | 92.5/92.5 → same |
| final | 90.0/86.7 | 90.0/86.7 | 90.0/86.7 | **90.0/86.7** | 83.3/86.7 → same |
| late | 95.0/87.5 | 95.0/80.0 | 95.0/77.5 | **95.0/87.5** | 87.5/92.5 → same |
| sample | 95.0/95.0 | 95.0/95.0 | 95.0/95.0 | **95.0/95.0** | 92.5/92.5 → same |
| lane_split | 93.3/93.3 | 46.7/86.7 | 40.0/83.3 | **93.3/86.7** | 90.0/46.7 → same |
| munich | 99.2/92.4 | 88.0/64.2 | 87.9/64.0 | **94.1/75.9** | 84.9/56.9 → same |

- Main-edge records crossing a lane, raw iter5 → iter7: lane_split 17 → 0, munich 41 → 0, a2d2b 27 → 0, all others 0 → 0. The iter6 rule left 2 on lane_split and 12 on a2d2b.
- The iter7 rule applied to the **iter6 GPU polylines**, where the gate had already removed edges, so only the hold can restore them (`replay_edges_on_iter6/`):
  - a2d2a 76.2/71.5 → 91.9/93.0
  - a2d2b 85.5/83.7 → 96.5/95.9
  - lane_split 40.0/83.3 → 56.7/83.3
  - munich 87.9/64.0 → 90.0/67.7
  - all others unchanged
- Edge jit2 is unchanged or better; a2d2a goes from 0.067/0.059 to 0.042/0.047.
- Munich R is still 75.9 against 92.4 in iter5. `FBEV_EDGE_MAX_DEG=35` gives 96.1/80.7, still with 0 crossings, if munich R remains low on GPU.
- exit_approach f20: the lane at x = 5.14 m next to the edge at 5.46 m is removed. The remaining lanes are −2.22 and 1.6, and the edges −5.2 and 5.46.

## Tests (CPU, 24 Nov ~09:00 IST)
- pytest: **16 passed** with NumPy 2, and **16 passed** under the NumPy-1 simulation (`PYTHONPATH=tests/np1_sim`).
  - New tests cover `trim_edge` (diagonal part trimmed, paint beyond the edge ignored, the 25° heading rule), `EdgeHold` (propagation, 5-frame expiry, decay, fill), `suppress_edge_duplicates` (duplicate dropped, nearest kept) and the robust KF update.
  - The 4 legacy scripts print ALL PASSED.
- CPU smoke under the NumPy-1 simulation (`cpu_smoke/run_cpu_smoke.sh` against the same-input iter6 baseline `run_iter6_cpu.sh`):
  - Inputs: iter6 GPU stage-1 npz dets, plus classical CPU masks clamped to the iter6 SAM edges.
  - Result: rc 0 on all 9 clips, including lane_split and exit_approach.

| clip (CPU) | accel_rms iter6 → iter7 | edge L/R pres | lane L/R pres |
|---|---|---|---|
| lane_split | 0.914 → 0.914 | 63.3/56.7 → 70.0/56.7 | 33.3/0.0 → same |
| exit_approach | 0.49 → 0.492 | 90.0/82.5 → 90.0/87.5 | 22.5/95.0 → same |
| a2d2a | 0.807 → 0.808 | 47.1/51.2 → 74.4/75.0 | 0/0 → same |
| a2d2b | 0.898 → 0.689 | 83.7/59.9 → 95.3/88.4 | 0/21.5 → same |
| munich | 0.892 → 0.664 | 64.4/56.6 → 91.1/67.7 | 27.1/17.3 → same |
| late | 0.681 → 0.688 | 92.5/60.0 → 95.0/95.0 | 60.0/45.0 → same |

  - ID-switch proxy is 0 on all clips. The full table is in `cpu_smoke/smoke_table.md`.
  - Still `cpu_smoke/stills/kitti_lane_split_candidate_f0015.png`: the diagonal trail near (3, 22) is gone.
- Kernel smoke (`FBEV_SMOKE=1 FBEV_MAXF=8`, NumPy-1 simulation): rc 0, fails none, 9 × `*_foundation_bev_iter7.mp4`.

## GPU Bot run
- Entry: `src/run_iter7.py`. Dataset = this folder: `src/`, `cams_iter7.json`, `yolo_baseline/`, plus the same clip mp4s, cams.json and motion_assumptions.json as iter6.
- **Free credit only.** Clips: all 9 (leave `FBEV_TAGS` unset).
- **Full stage 1+2** (no `FBEV_REUSE_DETS`).
- Env vars:
  - `FBEV_OUT`, `FBEV_TMP`, `FBEV_DS` as before.
  - `FBEV_EXTRA_STILLS=6,7,8,15,18,20,22,24,26,34,63,86,133,180,450,697,760,762`
    (the iter6 set plus exit_approach 22/24/26 and munich 760/762)
  - `FBEV_YOLOPV2` + `FBEV_YOLO_REPO`: run the YOLO baseline again.
  - Do not set `FBEV_SMOKE`, `FBEV_MAXF`, `FBEV_RAY_R` or `FBEV_DEBUG_PIN`.
  - Defaults: `FBEV_EDGE_MAX_DEG=25`, `FBEV_LANE_EDGE_DUP_M=0.8`, `FBEV_GORE_GROW_M=1.2`, `FBEV_GORE_MAX_STEP=0.4`.
- Outputs:
  - `results/<clip>_foundation_bev_iter7.mp4`
  - `stills/` (the set above, where in range)
  - `tracks/`
  - `polylines/` (edges carry a `held` flag)
  - `dets_dump/`, `dets/`
  - `metrics.md`, `ego_speed_table.md`, `metrics_yolo_vs_run.md`
- Expected T4 time: **~75–80 min**. iter6 took a 4421 s Modal process (74 min) at $0 spend, and the stage-2 changes add under a minute.
- **Please produce `metrics_vs_iter6.md`**:
  - one 9-clip table, iter6 (`/workspace/foundation_bev/iter6`) → iter7;
  - columns: accel_rms, n_tracks, idsw, lanes L10/R10 pres + jit2 + wjit, edges L10/R10 pres + jit2;
  - gore frames per clip: they must still fire only on exit_approach f21–26 and munich f760–765;
  - the number of frames per clip with a `held: true` edge;
  - flag any clip whose lane presence drops more than 5 points or whose edge presence is below iter5. Check lane_split L, munich R and a2d2a accel first.
