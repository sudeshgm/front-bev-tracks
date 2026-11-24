# iter11 SPEC: FINAL (24 Nov 2026, ~18:05 IST)
iter11 is iter10 plus five changes:
- a kerb-blend restart of the widen-gore history (fixes the false munich f881–885 gore);
- a veto of short diagonal edge fragments (recovers munich R f520–526 and f555–560);
- clip-start backfill of the main road edges;
- a 0.6 s clip-start window;
- the ego lane-pair width as a slow random walk, RTS-smoothed (`kf` mode).

Gating, association, relink, tracks, the trim rules, bridging, hold, the kerb rule, kerb ramp, the RHT dot rule and the centred smoothing are unchanged. There is no TwinLiteNet or YOLO in the pipeline; YOLOPv2 is only the measurement baseline. NumPy-1 compat and `FBEV_REUSE_DETS` still work.

`patch.diff` is a diff against `iter10_spec/src` (`lines_mt.py`, `foundation_bev.py`, `run_iter10.py` → `run_iter11.py`).

## Gore replay gate (`tools/replay_gore.py`, `tools/replay_iter11.sh`, `replay/gore_*.txt|json`)
- **What it replays:** the iter10 GPU `edges_raw` (untrimmed main edges) go through the in-step edge chain: `trim_edge` (with bridging) → `KerbState` (car boxes from `dets_dump`) → `EdgeHold` → `retrim_held` → `is_flared` → `DivergenceGore` → `GoreTracker`.
  - In iter11 this chain is `StageLines.post_edges`, split out of `step()`, so the replay runs the pipeline's own code.
  - In-step lanes were not dumped before iter11. The iter9 GPU lanes stand in for them; the lane tracker is upstream of every edge change.
  - The kerb traffic guard needs the in-step lanes. So car boxes are gated by the iter9 GPU kerb_occ frames (`--kerb-oracle`), and the kerb `_step` decisions follow the GPU while the ramp, bridging, hold and veto run for real.
- **It reproduces the GPU first**, on all 9 clips:
  - iter10 code (`replay/gore_iter10code_actual.txt`) and the iter11 code with the iter11 switches off (`gore_set10.txt`) give exactly the iter10 GPU widen gores: exit f21–26 and munich f881–885, nothing else.
  - The iter9 settings (`gore_set9.txt`) give exactly the iter9 GPU widen gores: exit f21–26 only.
  - The munich f760–765 gore is a lane-lane **fork** (child lane 569 of 565). It needs in-step lanes that `suppress_edge_duplicates` removed from the polylines, so the replay can't rebuild it. Its key (`("fork", side)`) and inputs (lanes, split edges) are untouched by iter11, and the CPU smoke keeps every fork frame.
  - `replay_iter11.sh` exits non-zero if the reproduction fails.
- **iter11 (`gore_iter11.txt`):** exit f21–26 still fire. Munich f881–885 is gone. No other clip fires.
- From iter11 on, the polylines carry `step` (in-step lanes before suppression, and split edges), so later replays need no lane proxy.

## Root cause of the munich f881–885 false widen gore
- The widen rule (iter6) fires when the gap between the nearest right lane line and main_R grows ≥ 1.2 m above its recent minimum, monotone, with each step ≤ `max_step` 0.4 m. A jump > 0.4 m restarts the gap history.
- On munich f866–890 main_R runs along a car row, so the kerb rule pushes it outward by up to 2.5 m:
  - **iter9:** the push switched on in one frame, the gap jumped by about 2.5 m, the history restarted, and there was no gore.
  - **iter10:** the 0.3 s kerb ramp blends the push in over `ramp_n` frames. The gap now climbs 3.76 → 4.00 → 4.27 → 4.53 → 4.90 m (+0.25–0.37 m per frame, under `max_step`). That is exactly the gradual monotone widening the rule looks for, so a candidate appears from f877, `n_persist` plus `GoreTracker` birth makes it fire f881–885, and it ends when the ramp-out restarts it.
- **Not the cause:**
  - The gap fill (bridging) does not touch f866–890.
  - The pair smoothing and backfill run after the gore pass.
- **Fix** (`DivergenceGore`, `FBEV_GORE_KERB_RESET=1`): every edge record carries `kerb_alpha` (the blend weight). A change of the weight restarts the gap history, the same as a jump. The edge model changed, not the road. Exit's real opening exit lane has no kerb and still fires.

## Changes
1. **Kerb-blend restart of the widen history** (above).
   - `StageLines.post_edges` (refactor, no behaviour change) and `kerb_alpha` on kerb edges, also dumped in the polylines.
2. **Diagonal fragment veto** (`is_diag_fragment`, in `post_edges` before the hold; `FBEV_EDGE_FRAG_M=6.0`, `FBEV_EDGE_FRAG_DEV=0.25`).
   - *Diagnosis:* on munich R f520–526 iter10's bridging plus trim left a 4 m piece at z 20.5–24.5 with slope 0.37. On f555–560 it left a 5.5 m piece at z 11.5–17 with slope 0.94. The road heading there is 0.03–0.07. Both are pieces of the image-border or forecourt hook (overlay `replay/overlays/munich_R_*.jpg`).
   - That live fragment replaced the line iter9's hold coasted. Where the hold had expired it was hold-filled into a "<" zigzag (f561–575, f603–609).
   - *Rule:* a live (not held, not kerb) main edge shorter than 6 m that starts beyond z 10 and deviates from the lane heading by more than 0.25 is not an edge for that frame, so the hold coasts as in iter9.
   - `frag_extend` (`FBEV_EDGE_FRAG_EXT_M=12`): a short far fragment that does run along the road heading is extended to z 10 along that heading, like the lane backfill, if `trim_edge` accepts it. It fired on no frame of the 9-clip replay; it is kept as the counterpart to the veto.
   - *Result (replay, munich R at z 10):*
     - recovered f520–526 and f555–560, plus f144–156 and f596–602;
     - lost f561–575 and f603–609, which were iter10's zigzags;
     - net +14 frames, presence 86.4 → 87.9.
   - f165–166 stay missing: there is no raw edge nearer than 22 m and the hold has expired. That isn't cheap to fix without a longer hold.
   - At z 10 these kerbs are at or beyond the right image border (u ≥ 1220 of 1280), so the recovered geometry is the coasted line, as in iter9.
3. **Edge backfill** (`edge_backfill`, after the lane backfill; `FBEV_EDGE_BACKFILL_S` defaults to the lane window).
   - The lane-backfill rule applied to main_L and main_R, keyed by role:
     - confirmed (≥ 4 frames) in the clip-start window;
     - copied back with the ego travel and its near slope;
     - re-trimmed against that frame's lanes;
     - marked `backfill: True`.
   - *Velocity correction* (`FBEV_EDGE_BACKFILL_CV=1`): the offset at z 10 follows the edge's own motion over its first live frames, clipped to ±0.5 m. The pure travel shift over-extrapolated diverging edges: on city the width steps were 0.31 m against 0.18 m live.
4. **Clip-start window 0.3 → 0.6 s** (`FBEV_LANE_BACKFILL_S=0.6`). final's ego-right lane is confirmed at f5 (0.5 s), and edges start at f4 (final R) and f5 (late R). The window is still clip-start only.
5. **Pair width as a slow random walk** (`pair_width_smooth` mode `kf`, the new default).
   - Per (left id, right id) run of ≥ 1 s, the offset of the ego pair's width profile is RTS-smoothed (Kalman forward pass plus Rauch–Tung–Striebel backward pass). The offset is the LS line of w(z) = x_R − x_L over z 6–25 m, evaluated at z 10. The model is a random walk with `FBEV_PAIR_KF_Q=0.1` m/√s and measurement noise `FBEV_PAIR_KF_R=0.15` m.
   - The profile's shape around that line is kept as measured. The change goes to the wobbling line (the iter10 `resid` split), with the iter10 fade (`max_dev` 0.5).
   - Shorter runs (lane_split's re-pairings during the split) keep iter10's `tri`. A local linear-trend model is available (`FBEV_PAIR_KF_MODEL=trend`), but it follows late's lane-change width drift (late 0.016).
   - The lane change and lane_split are unchanged:
     - final's lane change and lane_split are different pairs, so they are never smoothed across;
     - every lane_split lane metric is identical to iter10;
     - the largest line move is 0.20 m (late R), and final's is 0.05 m.
6. **Dump:** `step` per frame (in-step lanes and split edges), `kerb_alpha` and `frag` per edge record, and version `iter11`.

## Replay on the iter10 GPU outputs (`replay/replay_vs_iter10.md|json`)
- The baseline is "iter10 replay": the same harness with the iter11 switches off. It reproduces the iter10 GPU numbers: lanes exactly, and edges within 0.002 except lane_split R jit2.
- YOLOPv2 is the baseline only.

| clip | lane wjit YOLO · iter10 → iter11 | lane pres L/R iter10 → iter11 | lane jit2 L / R iter10 → iter11 | edge pres L/R iter10 → iter11 | edge wjit iter10 → iter11 |
|---|---|---|---|---|---|
| exit | 0.007 · 0.017 → **0.007** | 100/100 → 100/100 | 0.005/0.014 → 0.004/**0.008** | 95/95 → **100/100** | 0.048 → 0.047 |
| late | 0.009 · 0.019 → **0.006** | 97.5/100 → 97.5/100 | 0.011/0.007 → 0.011/0.011 | 95/87.5 → **100/100** | 0.035 → 0.041 |
| final | 0.008 · 0.012 → **0.006** | 93.3/90 → **100/100** | 0.015/0.004 → **0.011**/0.004 | 90/86.7 → **100/100** | 0.023 → 0.026 |
| sample | 0.006 · 0.004 → 0.002 | 100/100 | 0.003/0.004 → 0.003/0.004 | 95/95 → **100/100** | 0.021 → 0.022 |
| city | 0.005 · 0.004 → 0.001 | 100/100 | 0.002/0.003 → 0.002/0.001 | 93.3/93.3 → **100/100** | 0.054 → 0.076 |
| lane_split | 0.092 · 0.133 → 0.133 | 100/73.3 | 0.012/0.012 → same | 93.3/93.3 → **100/100** | 0.129 → 0.127 |
| munich | – · 0.006 → 0.006 | 85.9/65.4 | same | 99.8/86.4 → 100/**87.9** | 0.040 → 0.042 |
| a2d2a | – | 15.1/26.7 | same | 95.9/94.8 → 97.7/94.8 | 0.067 → 0.067 |
| a2d2b | – | 8.1/2.3 | same | 97.1/97.1 → 97.1/100 | 0.057 → 0.057 |

- **accel, n_tracks and idsw (0)** are identical; iter11 does not touch tracks.
- **No presence drops** anywhere.
- **Lane wjit is now at or below YOLO** on every KITTI highway and city clip. lane_split 0.133 vs 0.092 is unchanged (the `kf` prior would give 0.06 but costs lane_split jit2 0.012 → 0.05).
- **Edge wjit rises** on city (+0.022), late (+0.006), final (+0.003) and munich (+0.002).
  - On city, late and final the backfilled frames continue the clip-start width drift at the live rate (city 0.085–0.095 m per frame vs 0.09 live). It is the median now covering more drift frames, not new noise; with edge backfill off: city 0.054, late 0.035.
  - On munich it is the coasted line over the vetoed fragments.
- **late lane R jit2 0.007 → 0.011** (YOLO 0.014): during late's lane change the width is held, so the right line absorbs up to 0.2 m.

## Tests (CPU, 24 Nov ~18:00 IST)
- pytest: **52 passed** with NumPy 2 and **52 passed** under the NumPy-1 simulation. New tests (`tests/test_iter11.py`, 12 of them):
  - the kerb blend no longer fires a widen gore, and the iter10 switch reproduces the false gore;
  - a real exit opening still fires;
  - fragment veto cases, and the hold coasting over a vetoed fragment through `post_edges`;
  - `frag_extend`;
  - the RTS smoother;
  - `kf` beats `tri` on slow drift, keeps a pair switch, and falls back to `tri` on short runs;
  - edge backfill: clip start only, confirmation, no duplicates, lane crossing;
  - the 0.6 s window covers final's f5.
- **CPU smoke**, all 9 clips, NumPy-1 sim, iter10 GPU dets plus classical masks, iter10 code vs iter11 code (`cpu_smoke/cpu_smoke_iter10_vs_iter11.md`). rc 0 on all clips.
  - **accel, idsw, gores (frames and sources), held and kerb_occ are identical.**
  - Presence only goes up:
    - edges reach 100 on exit, late, sample and lane_split;
    - final lanes 76.7/80 → 90/100;
    - final lane wjit 0.011 → 0.007;
    - a2d2a edge wjit 0.088 → 0.092.
- **Kernel smoke** (`FBEV_SMOKE=1 FBEV_MAXF=8`, NumPy-1 sim): RC 0, fails none, all 9 clips version iter11 (`replay/kernel_smoke_tail.txt`).

## GPU Bot run
- Entry: `src/run_iter11.py`. Dataset = this folder: `src/`, `cams_iter11.json` (identical to iter10), `yolo_baseline/`, plus the same clip mp4s, cams.json and motion_assumptions.json as iter10.
- **Free credit only.** Clips: all 9 (leave `FBEV_TAGS` unset). **Full stage 1+2** (no `FBEV_REUSE_DETS`).
- Env vars:
  - `FBEV_OUT`, `FBEV_TMP`, `FBEV_DS` as before.
  - `FBEV_EXTRA_STILLS=0,1,2,3,4,5,6,7,8,15,17,18,19,20,21,22,23,24,25,26,27,28,29,34,60,63,80,86,100,107,119,121,123,133,140,146,160,170,180,240,258,266,267,300,450,520,523,526,545,557,560,565,587,603,625,647,677,692,697,700,760,762,878,879,880,881,882,883,884,885,886`
    - This is the iter10 set plus:
      - f0–5 on every clip (backfilled lanes and edges; final's R lane);
      - munich 520/523/526 and 557/560 (recovered R);
      - 565/603 (removed zigzag);
      - 146 (coast);
      - 878–886 (former false gore).
  - `FBEV_YOLOPV2` + `FBEV_YOLO_REPO`: YOLO baseline only, as before.
  - Do not set `FBEV_SMOKE`, `FBEV_MAXF`, `FBEV_WORLD_PIN`, `FBEV_LINE_SMOOTH_POSE`, `FBEV_TRIM_DEBUG`, `FBEV_RAY_R`, `FBEV_REPIN` or `FBEV_ROBUST_NIS`.
  - New defaults, so no need to set them: `FBEV_GORE_KERB_RESET=1`, `FBEV_EDGE_FRAG_M=6.0`, `FBEV_EDGE_FRAG_DEV=0.25`, `FBEV_EDGE_FRAG_EXT_M=12`, `FBEV_LANE_BACKFILL_S=0.6`, `FBEV_EDGE_BACKFILL_S` (= lane window), `FBEV_EDGE_BACKFILL_CV=1`, `FBEV_PAIR_W_MODE=kf`, `FBEV_PAIR_KF_MODEL=rw`, `FBEV_PAIR_KF_Q=0.1`, `FBEV_PAIR_KF_R=0.15`, `FBEV_PAIR_KF_MIN_S=1.0`.
  - iter10 defaults are unchanged: `FBEV_EDGE_GAP_M=5.0`, `FBEV_EDGE_GAP_SLOPE=0.5`, `FBEV_PAIR_W_S=1.5`, `FBEV_PAIR_W_MAXDEV=0.5`, `FBEV_KERB_RAMP_S=0.3`, kerb, RHT, `FBEV_LINE_SMOOTH_S=0.3`, `FBEV_EDGE_DIVERGE_DEG=45`, `FBEV_EDGE_HOLD_S=0.5`, etc.
- Outputs: as iter10 (`results/<clip>_foundation_bev_iter11.mp4`, `stills/`, `tracks/`, `polylines/` with the new `step` dump, `dets_dump/`, `dets/`, `metrics.md`, `ego_speed_table.md`, `metrics_yolo_vs_run.md`).
- Expected T4 time: **~75–80 min**, the same as iter10 (the new passes are CPU and cost seconds; the `step` dump adds about 15 % to the polylines JSON size (CPU smoke: 14.9 → 17.1 MB)).
- **Please produce `metrics_vs_iter10.md`:**
  - one 9-clip table, iter10 (`/workspace/foundation_bev/iter10`) → iter11. Columns: accel_rms, n_tracks, idsw, lanes L10/R10 pres + jit2 + wjit, edges L10/R10 pres + jit2 + wjit;
  - **gore frames per clip:** they must fire only on exit_approach f21–26 and munich f760–765, and **not** on munich f881–885;
  - held-edge, kerb_occ, frag and backfilled-edge frames per clip;
  - munich R at z 10: frames with a final edge. Check f520–526 and f557–560 (expected present), f561–575 (iter10 zigzag, expected absent) and f165–166 (still missing);
  - frame 0 on every clip: lanes and road edges present;
  - flag any clip whose lane or edge presence drops by more than 5 points vs iter10, any accel rise above 0.05, or idsw above 0.
- Caveat: stage 1 is not deterministic between runs, so accel can move by about ±0.05 independently of this code. iter11 does not touch tracks.

## Remaining YOLOPv2 gap (lanes; YOLO has no road edges)
- **lane_split lane wjit:** 0.133 vs 0.092. The `kf` prior on its short re-pairing runs would give 0.06 but moves the split lines (jit2 0.012 → 0.05).
- **exit lane R jit2:** 0.008 vs 0.009, now below YOLO (iter10: 0.014).
- **Ahead of YOLO:**
  - lane presence on lane_split;
  - lane wjit on exit (tied), late, final, sample and city;
  - jit2 L on every KITTI clip;
  - jit2 R on final, late and lane_split;
  - track accel (about 5× smoother).
