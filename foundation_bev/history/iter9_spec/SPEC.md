# iter9 SPEC: FINAL (24 Nov 2026, ~14:00 IST)
iter9 is iter8 plus three changes, listed below: a kerb-behind-parked-cars edge rule, a right-hand-traffic dot rule, and zero-lag centred line smoothing. Gating, association, relink, tracks, trim/hold and the gore rules are unchanged. All other render rules are unchanged too. NumPy-1 compat and `FBEV_REUSE_DETS` still work. There is no TwinLiteNet or YOLO in the pipeline; YOLOPv2 is only the measurement baseline.
`patch.diff` is a diff against `iter8_spec/src`. It covers `lines_mt.py` and `foundation_bev.py`, and renames `run_iter8.py` to `run_iter9.py`.

## Diagnosis (iter8 GPU outputs, `/workspace/gpubot/foundation-bev-iter8/outputs_modal/run_ok`)
**(a) lane_split f29: the right edge runs diagonally in front of the parked cars, and the near-right lane is short.**
- The main edges are the boundary of the SAM drivable-area mask warped to BEV. Object boxes are not filled into that mask, so next to a parked row the "edge" is the cars' bottoms and shadows.
  - At f29, main_R runs from px (531,456) to (347,320), directly under the car boxes [350,234,452,372] and [307,237,397,334].
  - The real kerb is lane 63, px (555,475)→(446,348), at BEV x 3.9–4.3.
- On f28–29 the edge is a held copy from f27 (slope −0.28 vs −0.15 for the left edge). The hold shifts by travel only. The ego is yawing, so over f17–29 the right edge converges by 6–9° relative to the left.
- **The near-right green lane is short** because paint under the car boxes is masked out, and the parked cars hide the paint beyond about 12 m. Lane 76 runs x 2.5→1.6 and lane 63 runs z 6–12; both are held.
- Edges bounded by car bottoms are common, not a lane_split one-off (frames per side):

  | clip | L | R |
  |---|---|---|
  | munich | – | 612 |
  | a2d2a | 102 | 61 |
  | a2d2b | 62 | 21 |
  | city | 28 | – |
  | lane_split | 5 | 24 |
  | KITTI highways | 0 | 0 |

**(b) Short red trails on parked cars (lane_split f29 z 12–18 / z 25, munich f692 right).** These are not yaw artifacts.
- The VO heading psi does **not** fit the track motion. For every clip, the psi scale that makes static tracks static is ≤ 0 (lane_split −0.5 to −0.75). Using +1 (an ego-compensated world frame) makes the drift *larger*:
  - lane_split #14: v_w 7.2 vs v_g 3.0 m/s;
  - munich #61: 2.75 vs 2.76 m/s.
  The VO-psi world pin is therefore built but **off** (`FBEV_WORLD_PIN=0`).
- A per-frame yaw/travel fit to the tracks themselves (static-majority IRLS) was also rejected. On highways and a2d2a the moving traffic is the majority, so it pins real movers (a2d2a person #35 and bicycle #32, highway cars).
- **The real cause.** lane_split #14 "closes" at 0.82 m/frame, while the neighbouring parked cars #6 and #7 close at about 0.4–0.5 m/frame (ego 4.6 m/s).
  - Its WD3D z jumps 26.9 → 24.8 → 20.2 over f16–19. That is association and depth hopping along the occluded parked row, and the final smoother turns it into a steady backward velocity.
  - munich #90 is the same pattern: first seen at x 9.8, moving backward at 2.3 m/s.
  - munich #61 and #82 creep *forward* at 2.2–2.8 m/s. They are a moving queue, so their short trails are real motion.

**(c) munich R 84.3 (iter5 92.4).** There are 141 miss frames on the iter8 GPU run:
- 75 frames: main_R only starts beyond z 10 (f2–13, f144–162 …). The pre-trim geometry is not in the iter8 outputs. On the iter5 raw output, the iter8 rule keeps these from 9.5 m. iter9 now dumps `edges_raw` (untrimmed main edges) to settle this on GPU.
- 41 frames: there was no edge even in iter5 (f189–211).
- **21 frames: trimmed at 45° and kept at 60° (f232–251).** The overlay at f240 shows this "edge" running along the image bottom across an open forecourt. It is junk, which confirms the corner-flare risk.
- 4 frames: trimmed even at 60°.
- **Decision: keep `FBEV_EDGE_DIVERGE_DEG=45`.** 60° would buy at most 21 frames (+2.3 points), and those are exactly the flares.

**(d) YOLO gap** (`replay/replay_vs_iter8.md`, YOLO rescored at the capture 10 Hz on the same frames): see the table at the end.

## Changes
1. **Kerb behind parked cars** (`lines_mt.car_footprints`, `kerb_behind_cars`, `kerb_apply`, `KerbState`; `FBEV_KERB_CARS=1`).
   - Pass 1 keeps the image car boxes per frame. Each box bottom is lifted to the ground, giving the contact row zb, the flanks and the central 70 %.
   - A main edge is **bounded** by a car if at least 2 of its samples within [zb−0.5, zb+1.5] fall inside that central part (the edge runs along the car's bottom).
   - Constraints come from cars on that side whose outer flank lies outside the edge by at most 2.5 m (`FBEV_KERB_MAX_PUSH_M`).
   - **Traffic guard:** a car is skipped when a long (≥ 8 m) painted lane line runs alongside it, at least 1 m inside its outer flank. This covers queues in the next lane (munich f692/f760). lane_split's bay line is short, so it does not count.
   - **Kerb line:** a straight line with the road heading (the opposite edge's slope for z ≤ 25, else the median lane heading) that clears every constraining car's outer flank (`FBEV_KERB_MARGIN_M=0`).
   - The edge becomes max(edge, line), capped at 2.5 m of push, up to the last car + 4.5 + 3 m, then ramped out over 3 m.
   - Only existing samples move, and only outward, so presence and z-extent are unchanged.
   - **Hysteresis** (`KerbState`):
     - enter after 3 consecutive bounded frames (`FBEV_KERB_ENTER`);
     - line = median (offset at z 10, slope, end) of the last 3 bounded lines, each carried with ego travel;
     - hold 5 frames (`FBEV_KERB_HOLD`), but only after at least 5 bounded frames.
     The per-frame version toggled with box flicker: a2d2a edge-L jit2 went to 0.64 on the CPU smoke.
   - Applied before the iter7/8 hold, so held edges inherit it. Edges get `kerb_occ: true` in polylines.
2. **RHT dot rule** (render only, `draw_bev2`; `FBEV_RHT_PIN=1`, `FBEV_RHT_MAX_RATIO=1.5`).
   - A track is drawn as a dot at its current position, with no trail, when all of these hold:
     - it was first seen right of the ego (x > 1.5 m);
     - it moves backward in the world (against the ego heading);
     - its motion is mostly longitudinal;
     - its speed is ≤ 1.5 × the ego speed.
   - Under right-hand traffic (all 9 clips: Germany / Italy) such a track cannot be oncoming traffic.
   - Left-side tracks keep their trails, because they can be oncoming. Example: lane_split #17 at z 23–25.
   - The iter6/7/8 rules (v_g < 1.5 → median pin; perpendicular → dot; young → dot) are unchanged and run first.
3. **Zero-lag centred line smoothing** (`lines_mt.centre_smooth_lines`; `FBEV_LINE_SMOOTH_S=0.3`, `FBEV_LINE_SMOOTH_SYM=1`, `FBEV_LINE_SMOOTH_POSE=0`).
   - After pass 2b (the whole clip is known before rendering), every tracked lane (by id) and main/split edge (by role+id) is averaged with its own copies in the neighbouring frames, using triangular weights.
   - Neighbour frames are moved into frame k by the selected travel s. This is the same convention the lane and edge holds use. The pose variant uses the VO psi, which bent lane_split lanes R (0.025 → 0.094).
   - **Symmetric window only:** at a track's birth or death the window shrinks to the frames available on both sides, so a one-sided average cannot bend a trend.
   - Only samples present in frame k change. Presence, z-extent, ids, roles, flags and fallback lanes are untouched.
   - Gores are computed during pass 2b, *before* this step, so smoothing cannot change gores.
4. **Polylines dump.** Each frame gains `edges_raw` (untrimmed main edges per role) and `ego` {x, z, psi}. Edge records gain `flare` (the iter8 SPEC said this was dumped, but it never was) and `kerb_occ`.

## Replay on the iter8 GPU outputs (`replay/replay_vs_iter8.md|json`)
`tools/replay_smooth.py <run_ok> cpu_smoke <out> --kerb` applies the kerb rule to the GPU main edges, using the same WD3D car boxes as the CPU run, then the smoothing. Tracks are unchanged, so **accel and idsw equal iter8** (lane_split 0.951, munich 0.655, a2d2a 0.495, a2d2b 0.821, late 0.534; idsw 0).

| clip | edge L/R pres | lane L/R pres | edge jit2 L/R iter8 → iter9 | lane jit2 L/R iter8 → iter9 | lane wjit |
|---|---|---|---|---|---|
| exit | 95/95 = | 95/95 = | 0.019/0.033 → 0.014/0.024 | 0.007/0.018 → 0.004/0.018 | 0.033 → 0.023 |
| sample | 95/95 = | 95/95 = | 0.026/0.018 → 0.017/0.013 | 0.007/0.005 → 0.005/0.006 | 0.007 → 0.006 |
| late | 95/87.5 = | 92.5/95 = | 0.024/0.040 → 0.018/0.020 | 0.012/0.019 → 0.011/0.008 | 0.020 → 0.020 |
| final | 90/86.7 = | 93.3/90 = | 0.033/0.040 → 0.030/0.025 | 0.039/0.006 → 0.013/0.004 | 0.024 → 0.013 |
| city | 93.3/93.3 = | 93.3/93.3 = | 0.065/0.026 → 0.021/0.010 | 0.004/0.010 → 0.002/0.006 | 0.008 → 0.007 |
| lane_split | 93.3/93.3 = | 93.3/73.3 = | 0.069/0.077 → 0.040/0.037 | 0.063/0.025 → 0.011/0.016 | 0.136 → 0.136 |
| munich | 99.8/84.3 = | 85.7/65.3 → 85.7/65.2 | 0.034/0.042 → 0.010/0.023 | 0.009/0.003 → 0.003/0.002 | 0.010 → 0.006 |
| a2d2a | 97.1/94.8 = | 16.9/23.3 → 15.1/25.0 | 0.044/0.047 → 0.041/0.040 | 0.001/0.010 → 0.008/0.008 | – |
| a2d2b | 97.1/97.1 = | 8.7/1.7 → 8.1/2.3 | 0.059/0.057 → 0.052/0.032 | 0.017/0.050 → 0.012/0.035 | – |

- **Edge width jitter** (edges_10_wjit) iter8 → iter9:

  | clip | iter8 → iter9 |
  |---|---|
  | exit | 0.055 → 0.048 |
  | sample | 0.027 → 0.020 |
  | late | 0.052 → 0.041 |
  | final | 0.033 → 0.023 |
  | city | 0.070 → 0.054 |
  | munich | 0.052 → 0.042 |
  | a2d2b | 0.085 → 0.085 |
  | lane_split | 0.107 → **0.130** |
  | a2d2a | 0.062 → **0.078** |

  The two increases come from the kerb rule switching on and off along parked rows. They are the price of putting the edge on the kerb.
- **Kerb-rule frames** (GPU replay): munich 225, a2d2a 109, a2d2b 33, lane_split 15, all others 0.
  - The overlays in `replay/kerb_overlays/` show the GPU iter8 edges (top) against the iter9 replay (bottom); red marks kerb edges.
  - lane_split f29: the right edge now runs behind the white car along the kerb.
  - munich f692/f760: the traffic queue is left alone.
  - munich f300: the edge sits behind the parked row.
  - a2d2a f60: the left edge sits behind the parked car.
- **Offline DivergenceGore on kerb-only edges: identical to iter8** (exit f19–22; munich f247 plus f587–608 raw fires, which the GoreTracker removes).
- **RHT replay on the GPU tracks** (`replay/replay_rht_gpu_iter8.txt`, `tools/replay_rht.py`):
  - lane_split: 16 trail track-frames become dots (tracks #3 and #14; #14 is the f17–29 diagonal);
  - munich: 27 (#90, f677–703, the f692 right-side diagonal);
  - all other clips: 0.

## Tests (CPU, 24 Nov ~13:40 IST)
- pytest: **30 passed** with NumPy 2 and **30 passed** under the NumPy-1 simulation. The 4 legacy scripts pass.
  - New tests (`tests/test_iter9.py`, 8 of them):
    - f29 car footprints;
    - the kerb is outward-only, keeps the z-extent and is capped;
    - no push when not bounded;
    - a traffic queue is skipped;
    - KerbState entry, and that a 2-frame burst is never applied or held;
    - smoothing is exact on static lines;
    - smoothing keeps a linear trend linear;
    - smoothing reduces noise.
- **CPU smoke**, all 9 clips, NumPy-1 sim: iter8 GPU dets plus classical masks, iter9 code vs iter8 code. rc 0 on all clips (`cpu_smoke/smoke_iter8_vs_iter9.md`).
  - **accel, idsw and n_tracks are identical on all 9 clips.**
  - Gores are identical, and held main-edge records are identical.
  - Presence changes by at most 0.2 points (munich edge R 91.0 → 90.8).
  - Edge jit2 is lower on most clips: munich L/R 0.042/0.046 → 0.017/0.029, late R 0.058 → 0.034, city R 0.042 → 0.023.
  - Small regressions:
    - lane_split lanes wjit 0.063 → 0.096 (classical-mask lanes; unchanged in the GPU replay);
    - a2d2a edge wjit 0.079 → 0.089.
  - Stills checked: the munich f692/f697 right-side diagonal trails are gone. On lane_split, the left-side #17 trail remains by design.
- **Kernel smoke** (`FBEV_SMOKE=1 FBEV_MAXF=8`, NumPy-1 sim): rc 0, fails none.

## GPU Bot run
- Entry: `src/run_iter9.py`. Dataset = this folder: `src/`, `cams_iter9.json` (identical to iter8), `yolo_baseline/`, plus the same clip mp4s, cams.json and motion_assumptions.json as iter8.
- **Free credit only.** Clips: all 9 (leave `FBEV_TAGS` unset).
- **Full stage 1+2** (no `FBEV_REUSE_DETS`).
- Env vars:
  - `FBEV_OUT`, `FBEV_TMP`, `FBEV_DS` as before.
  - `FBEV_EXTRA_STILLS=6,7,8,15,17,18,19,20,21,22,23,24,25,26,27,28,29,34,60,63,80,86,100,107,119,121,123,133,140,180,240,300,450,587,677,692,697,700,760,762`
    - This is the iter8 set plus the iter9 fix frames:
      - lane_split f17–29: kerb edge and the #14 dot;
      - munich 240 (flare), 300 (parked-row kerb), 677/692/697/700 (#90 dot, queue left alone), 760/762 (gore + queue);
      - a2d2a 60 (left kerb);
      - a2d2b 80.
  - `FBEV_YOLOPV2` + `FBEV_YOLO_REPO`: YOLO baseline only, as before.
  - Do not set `FBEV_SMOKE`, `FBEV_MAXF`, `FBEV_WORLD_PIN`, `FBEV_LINE_SMOOTH_POSE`, `FBEV_RAY_R`, `FBEV_REPIN` or `FBEV_ROBUST_NIS`.
  - New defaults, so no need to set them: `FBEV_KERB_CARS=1`, `FBEV_KERB_MAX_PUSH_M=2.5`, `FBEV_KERB_MARGIN_M=0`, `FBEV_KERB_ENTER=3`, `FBEV_KERB_HOLD=5`, `FBEV_RHT_PIN=1`, `FBEV_RHT_MAX_RATIO=1.5`, `FBEV_LINE_SMOOTH_S=0.3`, `FBEV_LINE_SMOOTH_SYM=1`.
  - iter8 defaults are unchanged: `FBEV_EDGE_MAX_DEG=30`, `FBEV_EDGE_DIVERGE_DEG=45`, `FBEV_EDGE_EGO_HALF_M=1.2`, `FBEV_EDGE_HOLD_S=0.5`, `FBEV_FAR_MODE=ray`, etc.
- Outputs:
  - `results/<clip>_foundation_bev_iter9.mp4`
  - `stills/`
  - `tracks/`
  - `polylines/` (now with `edges_raw`, `ego`, `flare`, `kerb_occ`)
  - `dets_dump/`, `dets/`
  - `metrics.md`, `ego_speed_table.md`, `metrics_yolo_vs_run.md`
- Expected T4 time: **~75–80 min**, the same as iter8 (the smoothing pass is CPU and costs seconds).
- **Please produce `metrics_vs_iter8.md`**:
  - one 9-clip table, iter8 (`/workspace/foundation_bev/iter8`) → iter9;
  - columns: accel_rms, n_tracks, idsw, lanes L10/R10 pres + jit2 + wjit, edges L10/R10 pres + jit2 + wjit;
  - gore frames per clip: they must fire only on exit_approach f21–26 and munich f760–765;
  - held-edge frames per clip;
  - kerb_occ edge frames per clip;
  - flag any clip whose lane or edge presence drops by more than 5 points vs iter8, any accel rise above 0.05, or idsw above 0.
  - Also please report munich R at z 10 for `edges_raw` vs the trimmed edges (how many of the frames where main_R starts beyond z 10 were trimmed, and how many were never there). That data settles (c).
- Caveat: stage 1 is not deterministic between runs, so accel can move by about ±0.05 independently of this code. iter9 does not touch tracks.

## YOLOPv2 gap (lanes; YOLO has no road edges; YOLO accel 3.1–4.7 vs ours 0.53–0.95 at 10 Hz)
| clip | lane pres L/R YOLO · iter9 | lane jit2 L YOLO · iter8 → iter9 | lane jit2 R | lane wjit |
|---|---|---|---|---|
| exit | 100/100 · 95/95 | 0.013 · 0.007 → 0.004 | 0.009 · 0.018 → 0.018 | 0.007 · 0.033 → 0.023 |
| sample | 100/100 · 95/95 | 0.019 · 0.007 → 0.005 | 0.005 · 0.005 → 0.006 | 0.006 · 0.007 → 0.006 |
| late | 100/100 · 92.5/95 | 0.014 · 0.012 → 0.011 | 0.014 · 0.019 → 0.008 | 0.009 · 0.020 → 0.020 |
| final | 100/100 · 93.3/90 | 0.024 · 0.039 → 0.013 | 0.017 · 0.006 → 0.004 | 0.008 · 0.024 → 0.013 |
| city | 100/100 · 93.3/93.3 | 0.001 · 0.004 → 0.002 | 0.004 · 0.010 → 0.006 | 0.005 · 0.008 → 0.007 |
| lane_split | 76.7/26.7 · 93.3/73.3 | 0.257 · 0.063 → 0.011 | 0.721 · 0.025 → 0.016 | 0.092 · 0.136 → 0.136 |

**Where we are still behind YOLO**
- **Lane presence on KITTI: 90–95 % vs 100 %.** Our misses are the first 1–2 frames of each clip, because a lane needs hits before it is born.
- **Lane width jitter (wjit):**

  | clip | ours | YOLO |
  |---|---|---|
  | exit | 0.023 | 0.007 |
  | late | 0.020 | 0.009 |
  | final | 0.013 | 0.008 |
  | city | 0.007 | 0.005 |
  | lane_split | 0.136 | 0.092 |

- **Right-lane jit2** on exit: 0.018 vs 0.009; city: 0.006 vs 0.004.

**Where we are level or ahead**
- Left-lane jit2 is now at or below YOLO on every KITTI clip.
- Final and late lane R are below YOLO.
- lane_split is far ahead.
- Tracks are about 5× smoother.
- YOLO draws no road edges at all.

The next candidates are lane birth on frame 0–1 (presence) and a joint L–R width prior (wjit).
