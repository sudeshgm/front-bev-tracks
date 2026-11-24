# iter6 SPEC: FINAL (24 Nov 2026 ~05:00 IST)
iter6 = iter5b plus the four fixes below. The NumPy-1 compat (`_trapz`) and its tests are kept. patch.diff is against iter5b_spec/src.

## Changes
1. **Parked-car sideways trails** (`foundation_bev.draw_bev2`).
   - A track with ≥5 rows whose *longitudinal* ground speed (selected ego travel) is below `STATIC_PIN_MPS = 1.5` m/s is pinned.
   - Pinned means drawn as one dot at its median world position, with no trail.
   - Lateral speed is ignored on purpose: the 5–8 m sideways drift *is* the artifact.
   - The 2 s / 15 m cap stays for moving tracks. This is drawing only; tracks.csv and the metrics are unchanged.
2. **Lane presence / false gores** (`lines_mt`).
   - `clean_lanes` drops a lane only if it is short (< 6 m) **and** young (< 4 hits). iter5 dropped it if either held.
   - The per-side cap is 4 (was 3), and the zigzag step limit is 0.5 m (was 0.3).
   - Stripe gores need 3 frames (GoreTracker n_birth 3).
   - Divergence gores go through the new `DivergenceGore`:
     - child-lane forks (and split edges) must persist ≥6 consecutive frames;
     - the single-frame lane-vs-edge wedge is removed, because it fired persistently on munich, a2d2 and city kerbs.
3. **exit_approach gore (temporal "widen" rule).**
   - Why it never fired on GPU: the exit lane's own outer line hugs the SAM road edge (gap ≈ 0), so the iter5 outermost-lane→edge wedge never grew.
   - The new rule uses the **nearest** lane line on each side. It tracks the median gap from that line to the side's main edge over z 6–20 m across frames.
   - It fires when the gap is ≥1.2 m above the recent minimum, rose monotonically over the last 3 steps, and rose gradually: every step ≤0.4 m, and a jump restarts the history.
   - Knobs: `FBEV_GORE_GROW_M`, `FBEV_GORE_MAX_STEP`.
   - Replayed offline on the **iter5 GPU polylines** (`tools/check_divergence.py`): exit_approach fires on f19–f22 (the gap grows 2.0 → 4.0 m over f10–f22). The other 8 clips fire 0 frames, except munich (f247, plus a child-lane fork around f587).
4. **Diagonal road edges.**
   - A main edge is dropped if its heading over z 5–15 m is more than 15° off the *road* heading (`FBEV_EDGE_MAX_DEG`). Road heading is the median lane heading, or the ego heading if there are no lanes.
   - A main edge is also dropped if it crosses a tracked lane line, meaning ≥3 m clearly (>0.5 m) on each side.
   - On iter5 GPU polylines this would drop:
     - lane_split: 14 crossing + 2 heading of 56 main-edge records;
     - exit_approach: 0;
     - **munich: 54 crossing + 347 heading of 1727**. Munich edge presence may drop; if it does, set `FBEV_EDGE_MAX_DEG=25`.

## Tests (CPU, 24 Nov ~04:55 IST)
- pytest: 12 passed with NumPy 2.5.5, and 12 passed under the NumPy-1 simulation (`PYTHONPATH=tests/np1_sim`).
  New tests cover the loosened clutter rules, `heading_ok`, `crosses_lane`, and `DivergenceGore` (gradual widening fires after ≥4 frames; flicker and a 1.5 m jump do not).
  The 4 legacy test scripts print ALL PASSED.
- `cpu_smoke/run_cpu_smoke.sh` (NumPy-1 simulation; iter5 dets_dump; classical CPU masks clamped to the iter5 SAM edges) on lane_split and exit_approach: rc 0.
  `cpu_smoke/run_iter5b_cpu.sh` gives the same-input iter5b baseline.

| clip (CPU replay) | lane L/R pres % | edge L/R pres % | edge L/R jit2 | gore frames |
|---|---|---|---|---|
| lane_split, iter5b | 26.7/36.7 | 86.7/73.3 | .029/.046 | 9 (stripes 2–7, 27–29) |
| lane_split, iter6 | **46.7**/33.3 | 70.0/60.0 | **.013**/.046 | 5 (stripes 3–7) |
| exit_approach, iter5b | 22.5/92.5 | 90.0/95.0 | .029/.048 | 7 (fork_edge 4–10) |
| exit_approach, iter6 | 22.5/95.0 | 90.0/82.5 | .029/.045 | 0 on CPU masks; f19–22 on the iter5 GPU polylines |

- Side-by-side BEV (`cpu_smoke/sbs_lane_split_f0015_bev.png`): the parked-car sideways trails are gone in iter6; one moving car keeps a short trail.
- Kernel smoke (`FBEV_SMOKE=1 FBEV_MAXF=8`, NumPy-1 simulation): rc 0, 9 × `*_foundation_bev_iter6.mp4`.

## GPU Bot run
- Entry: `src/run_iter6.py`. Dataset = this folder: `src/`, `cams_iter6.json`, `yolo_baseline/`, plus the same clip mp4s, cams.json and motion_assumptions.json as iter5.
- Clips: all 9 (leave `FBEV_TAGS` unset).
- Env vars:
  - `FBEV_OUT`, `FBEV_TMP`, `FBEV_DS` as before.
  - `FBEV_EXTRA_STILLS=6,7,8,15,18,20,34,63,86,133,180,450,697`
  - `FBEV_YOLOPV2` + `FBEV_YOLO_REPO`: run the YOLO baseline again.
  - Do not set `FBEV_SMOKE` or `FBEV_MAXF`.
  - Optional knobs: `FBEV_EDGE_MAX_DEG` (15), `FBEV_GORE_GROW_M` (1.2), `FBEV_GORE_MAX_STEP` (0.4).
- Stage 1: the volumes were deleted, so this is a **full stage 1+2** run (no `FBEV_REUSE_DETS`).
  - Option, only if the user approves: `FBEV_REUSE_DETS=<copy of /workspace/foundation_bev/iter5/dets_dump>` works for all 9 clips via the `<tag>_dets_ego.json` fallback.
  - That dump keeps object scores and 3D boxes, but lane/edge prompt boxes reload with a fixed 0.5 score and a single class each, so stage 2 is not bit-identical to stage 1 output.
- Outputs:
  - `results/<clip>_foundation_bev_iter6.mp4`
  - `stills/` with the set above:
    - KITTI f0006/7/8/15/18/20 where in range
    - munich f0180/0450/0697
    - a2d2 f0034/0063/0086/0133
  - `tracks/`, `polylines/` (gores carry `src`: stripes / fork / widen), `dets_dump/`, `metrics.md`, `ego_speed_table.md`, `metrics_yolo_vs_run.md`
- Expected T4 time: **~90 min** for the full run (iter5: stage 1 56 min + stage 2 ~20 min + YOLO 3.5 min + installs). With dets reuse: ~25 min.
- **Please produce `metrics_vs_iter5.md`**:
  - one 9-clip table, iter5 (merged iter5 + iter5b, at /workspace/foundation_bev/iter5) → iter6;
  - columns: accel_rms, n_tracks, lanes L10/R10 pres + jit2 + wjit, edges L10/R10 pres + jit2;
  - plus gore frames per clip.
  - Flag any clip whose lane or edge presence drops > 5 points; check munich edges first.
