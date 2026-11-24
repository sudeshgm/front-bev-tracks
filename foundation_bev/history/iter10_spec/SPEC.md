# iter10 SPEC: FINAL (24 Nov 2026, ~15:49 IST)
iter10 is iter9 plus four changes: edge gap bridging before trim, ego lane-pair width smoothing, clip-start lane backfill and a kerb blend in/out. Gating, association, relink, tracks, the trim rules themselves, hold, gores, the kerb rule, the RHT dot rule and the centred smoothing are unchanged. There is no TwinLiteNet or YOLO in the pipeline; YOLOPv2 is only the measurement baseline. NumPy-1 compat and `FBEV_REUSE_DETS` still work.
`patch.diff` is a diff against `iter9_spec/src` (`lines_mt.py`, `foundation_bev.py`, `run_iter9.py` → `run_iter10.py`).

## Diagnosis: munich R at z = 10 (iter9 GPU `edges_raw`, `replay/munich_R_trim_attr.txt`, `tools/trim_gap_attr.py`)
**The 75 "trim-killed" frames are two different things:**
- **Holes in the raw edge, then the nearest-run rule (40 frames).** `edges_raw` has interior NaN holes of 1–6.5 m where parked or stopped cars and shadows break the SAM boundary. `trim_edge` keeps the *nearest* good run of at least 4 m, so a 1.5 m near piece plus a hole made it skip to the far part.
  - f2–7: 9.5–10.5, then a hole to 17 → kept 21–44.
  - f258–267: 9.5–10.5, a hole at 11.0, then 11.5–15, then a 15.5 m hole → kept 11.5–15 (the "3 m edge" on f266).
  - f158–163: a hole across z = 10.
  - No lane-crossing, ego-corridor or heading sub-rule is involved in these frames. My earlier replay interpolated `edges_raw` across the holes, which hid this.
- **The 45° divergence rule (59 frames: f228–254, 535–556, 588–602, 612–618).** The overlays (f240, f545, f595, f614) show forecourt and side-entry flares running to the image corner with no kerb near the ego. **These stay trimmed** (`FBEV_EDGE_DIVERGE_DEG=45` unchanged).
- The rest: 62 frames have no raw main_R (f164–227, the forecourt) and 10 start beyond z 10 (f148–157). These can't be fixed by trimming.

## Changes
1. **Edge gap bridging before trim** (`lines_mt.bridge_gaps`, called first in `trim_edge`; `FBEV_EDGE_GAP_M=5.0`, `FBEV_EDGE_GAP_SLOPE=0.5`).
   - Interior holes up to 5 m are filled linearly when the bridge runs with the road (|slope − median lane slope| ≤ 0.5, about 27°). Leading and trailing NaN are never extrapolated.
   - The lane-crossing, ego-corridor and heading rules then judge the bridged samples like measured ones, so flares and crossings stay out.
   - The 5 m and 0.5 limits were set from overlays:
     - a 6.5 m hole on f2–7 bridged into a diagonal hook in front of the parked cars, so the limit is 5 m;
     - slope 1.0 let the f8–12 and f144–147 forecourt hooks back in, so the limit is 0.5.
   - Optional diagnosis: `FBEV_TRIM_DEBUG=<file.jsonl>` writes each trim's masks, lanes and kept run (off by default).
2. **Ego lane-pair width smoothing** (`lines_mt.pair_width_smooth`, run after the centred smoothing; `FBEV_PAIR_W_S=1.5`, `FBEV_PAIR_W_MAXDEV=0.5`, `FBEV_PAIR_W_SPLIT=resid`, `FBEV_PAIR_W_MODE=tri`).
   - The ego pair is the nearest left and right tracked lane at z = 10 (the metrics' definition). It is smoothed per run of frames with the same (left id, right id).
   - The width profile w(z) = x_R − x_L is averaged over ±1.5 s with triangular weights, symmetric and clipped to the run. Neighbours are moved by the ego travel, as in the centred smoothing.
   - The change goes to the line that wobbles: one weight per run, from each line's RMS offset from its own trend. Exit's wobble is the right line pumping 1.16 ↔ 1.98 m with the dash phase.
   - Changes above 0.25 m fade out, reaching zero at 0.75 m, so a real split or merge is left alone. Presence, ids and z-extent are unchanged.
   - Tested and rejected: a constant-width prior per run (`mean`/`median`/`auto`). It cut lane_split wjit 0.136 → 0.06 but cost the final lane change (wjit 0.013 → 0.018–0.020, L jit2 0.013 → 0.022–0.028).
3. **Clip-start lane backfill** (`lines_mt.lane_backfill`; `FBEV_LANE_BACKFILL_S=0.3`).
   - A lane confirmed in the first 0.3 s (still tracked 4 frames later) is copied back into the earlier frames, moved by the ego travel. Where that leaves z = 10 uncovered, its near end is extended along its near slope (by at most the travel).
   - No copy is made where a lane already lies within 0.8 m. Mid-clip births are never backfilled.
   - Records get `backfill: true` in polylines.
   - 0.5 s would also cover final, but it raised final L jit2 0.015 → 0.020.
4. **Kerb blend in/out** (`KerbState(fps)`; `FBEV_KERB_RAMP_S=0.3`, 0 = the iter9 switch).
   - The push ramps in over 0.3 s after entry. After the kerb line is dropped, the last line, carried with the ego travel, ramps out over 0.3 s instead of snapping back.
5. **Polylines dump:** the records and `*_bev` lists are kept aligned (a polyline with fewer than 2 points used to be dropped from `*_bev` only). Lane records gain `backfill`.

## Replay on the iter9 GPU outputs (`replay/replay_vs_iter9.md|json`, `tools/replay_iter10.sh`)
- The lanes are the unsmoothed iter8 GPU lanes run through the pipeline functions. The edges are the iter8 GPU edges plus KerbState and smoothing.
- The baseline is "iter9 replay" (the same harness with the iter10 parts off). It reproduces the iter9 GPU numbers: lane metrics exactly, edges within 0.005 except a2d2b.
- Tracks are untouched, so **accel and idsw equal iter9** (lane_split 0.951, munich 0.655, a2d2a 0.495, a2d2b 0.821; idsw 0 everywhere).

| clip | lane pres L/R YOLO · iter9 → iter10 | lane wjit YOLO · iter9 → iter10 | lane jit2 L YOLO · iter9 → iter10 | lane jit2 R YOLO · iter9 → iter10 | edge wjit iter9 → iter10 |
|---|---|---|---|---|---|
| exit | 100/100 · 95/95 → **100/100** | 0.007 · 0.023 → **0.017** | 0.013 · 0.005 → 0.005 | 0.009 · 0.018 → **0.014** | 0.048 = |
| sample | 100/100 · 95/95 → **100/100** | 0.006 · 0.006 → **0.004** | 0.019 · 0.005 → 0.003 | 0.005 · 0.004 → 0.004 | 0.019 = |
| late | 100/100 · 92.5/95 → **97.5/100** | 0.009 · 0.020 → 0.019 | 0.014 · 0.012 → 0.011 | 0.014 · 0.008 → 0.007 | 0.036 = |
| final | 100/100 · 93.3/90 → 93.3/90 | 0.008 · 0.013 → 0.012 | 0.024 · 0.013 → 0.015 | 0.017 · 0.004 → 0.004 | 0.023 = |
| city | 100/100 · 93.3/93.3 → **100/100** | 0.005 · 0.007 → **0.004** | 0.001 · 0.002 → 0.002 | 0.004 · 0.006 → **0.003** | 0.054 = |
| lane_split | 76.7/26.7 · 93.3/73.3 → **100/73.3** | 0.092 · 0.136 → 0.133 | 0.257 · 0.011 → 0.012 | 0.721 · 0.016 → 0.012 | 0.129 = |
| munich | – · 85.7/65.2 → 85.9/65.4 | – · 0.006 = | – · 0.003 = | – · 0.002 = | 0.042 → **0.038** |
| a2d2a | – · 15.1/25.0 → 15.1/26.7 | – | – · 0.008 = | – · 0.007 = | 0.075 → **0.056** |
| a2d2b | – · 8.1/2.3 = | – | – · 0.012 = | – · 0.035 = | 0.085 → **0.079** |

**Target-by-target:**
- **(a) munich R** (`replay/replay_gap.txt`):
  - Gap bridging recovers **26 frames at z = 10**: f158–163, 167–173, 256–267 (including **f266–267**) and 647.
  - Zero losses, and no gained frame is flared.
  - Estimated munich edge R ~84.1 → ~87.0 on the GPU metric (+26/900). The edge replay above cannot show this, because the trim runs before hold and kerb.
  - All other clips: no z = 10 change. Bridging touches only city main_L f25–29 (near end 9.0 → 8.0 m) and munich R.
  - Munich also gets longer far geometry on f618–645 (16.5 → 44.5 m along the kerb, checked on overlays).
  - Not recovered, by design: f2–12 and f144–147 (hooks in front of parked cars or into the forecourt), the 59 flare frames, and the 72 frames with no raw edge or a raw edge starting beyond z 10.
- **(b) lane width jitter:**
  - exit 0.023 → 0.017 (YOLO 0.007);
  - city 0.007 → 0.004 (better than YOLO's 0.005);
  - sample 0.006 → 0.004;
  - final 0.013 → 0.012;
  - late 0.020 → 0.019;
  - right-lane jit2: exit 0.018 → 0.014, city 0.006 → 0.003.
  - The remaining exit, late and final gap is a slow drift over 1–3 s (late: width 3.85 → 3.2 within one pair), not frame noise. A centred window keeps linear trends, and a constant-width prior breaks the final lane change.
- **(c) lane presence:** exit, sample and city go to 100/100, late to 97.5/100 and lane_split L to 100.
  - final stays 93.3/90: its right lane is born at f5, after the 0.3 s window.
- **(d) edge wjit:** a2d2a 0.075 → 0.056 (below iter8's 0.062), munich 0.042 → 0.038, a2d2b 0.085 → 0.079.
  - lane_split stays at 0.129. There the kerb is on continuously from f15 to f29, so it is not switching. Its wjit is the real geometry: the left split edge diverges from −2.6 to −5.7 m while the right kerb converges. Its R jit2 improves 0.037 → 0.034.
- **(e) a2d2a lane L 16.9 → 15.1:** this is not a lost lane. On f25, f27 and f73 the ego straddles the lane (x ≈ 0 at z 10), and the smoothing nudges it ±0.05 m across 0, so the metric counts it as R. Left as is.
- **Gores:** bridging changes nothing after munich f647 and nothing on exit, so the exit f21–26 and munich f760–765 gores are unaffected. The pair smoothing and backfill run after the gore pass.

## Tests (CPU, 24 Nov ~15:49 IST)
- pytest: **40 passed** with NumPy 2 and **40 passed** under the NumPy-1 simulation. New tests (`tests/test_iter10.py`, 10 of them):
  - bridging fills only short interior holes, never extrapolates, rejects steep bridges, and recovers the f258 pattern;
  - flares and crossings stay out;
  - pair smoothing reduces the wobble on the wobbling line only and keeps a pair switch;
  - backfill applies at the clip start only, without duplicates;
  - the kerb ramps in and out, and ramp 0 = iter9.
- **CPU smoke**, all 9 clips, NumPy-1 sim, iter9 GPU dets plus classical masks, iter10 code vs iter9 code (`cpu_smoke/cpu_smoke_iter9_vs_iter10.md`). rc 0 on all clips.
  - **accel, idsw and gores are identical.**
  - Presence only goes up: a2d2a edge R 82.0 → 89.5, munich edge R 73.6 → 75.1, city edge L 76.7 → 80.0, plus backfilled lanes.
  - Small rises on the classical-mask lanes: exit lane wjit 0.009 → 0.011, late 0.009 → 0.012, munich edge wjit 0.058 → 0.061.
- **Kernel smoke** (`FBEV_SMOKE=1 FBEV_MAXF=8`, NumPy-1 sim): RC 0, fails none, all 9 clips version iter10 (`replay/kernel_smoke_tail.txt`).

## GPU Bot run
- Entry: `src/run_iter10.py`. Dataset = this folder: `src/`, `cams_iter10.json` (identical to iter9), `yolo_baseline/`, plus the same clip mp4s, cams.json and motion_assumptions.json as iter9.
- **Free credit only.** Clips: all 9 (leave `FBEV_TAGS` unset). **Full stage 1+2** (no `FBEV_REUSE_DETS`).
- Env vars:
  - `FBEV_OUT`, `FBEV_TMP`, `FBEV_DS` as before.
  - `FBEV_EXTRA_STILLS=0,1,2,6,7,8,15,17,18,19,20,21,22,23,24,25,26,27,28,29,34,60,63,80,86,100,107,119,121,123,133,140,160,170,180,240,258,266,267,300,450,545,587,625,647,677,692,697,700,760,762`
    - This is the iter9 set plus:
      - f0–2 (backfilled lanes on every clip);
      - munich 160/170/258/266/267/647 (recovered right edge), 545 (flare must stay trimmed) and 625 (longer far kerb).
  - `FBEV_YOLOPV2` + `FBEV_YOLO_REPO`: YOLO baseline only, as before.
  - Do not set `FBEV_SMOKE`, `FBEV_MAXF`, `FBEV_WORLD_PIN`, `FBEV_LINE_SMOOTH_POSE`, `FBEV_TRIM_DEBUG`, `FBEV_RAY_R`, `FBEV_REPIN` or `FBEV_ROBUST_NIS`.
  - New defaults, so no need to set them: `FBEV_EDGE_GAP_M=5.0`, `FBEV_EDGE_GAP_SLOPE=0.5`, `FBEV_PAIR_W_S=1.5`, `FBEV_PAIR_W_MAXDEV=0.5`, `FBEV_PAIR_W_SPLIT=resid`, `FBEV_PAIR_W_MODE=tri`, `FBEV_LANE_BACKFILL_S=0.3`, `FBEV_KERB_RAMP_S=0.3`.
  - iter9 defaults are unchanged: kerb, RHT, `FBEV_LINE_SMOOTH_S=0.3`, `FBEV_EDGE_DIVERGE_DEG=45`, `FBEV_EDGE_MAX_DEG=30`, `FBEV_EDGE_EGO_HALF_M=1.2`, `FBEV_EDGE_HOLD_S=0.5`, etc.
- Outputs: as iter9 (`results/<clip>_foundation_bev_iter10.mp4`, `stills/`, `tracks/`, `polylines/`, `dets_dump/`, `dets/`, `metrics.md`, `ego_speed_table.md`, `metrics_yolo_vs_run.md`).
- Expected T4 time: **~75–80 min**, the same as iter9 (the new passes are CPU and cost seconds).
- **Please produce `metrics_vs_iter9.md`:**
  - one 9-clip table, iter9 (`/workspace/foundation_bev/iter9`) → iter10. Columns: accel_rms, n_tracks, idsw, lanes L10/R10 pres + jit2 + wjit, edges L10/R10 pres + jit2 + wjit;
  - gore frames per clip: they must fire only on exit_approach f21–26 and munich f760–765;
  - held-edge frames and kerb_occ edge frames per clip;
  - munich R at z 10: frames with a final edge vs `edges_raw` (iter9: 757 vs 824). The expected gain is about +26 (f158–173, f256–267, f647);
  - flag any clip whose lane or edge presence drops by more than 5 points vs iter9, any accel rise above 0.05, or idsw above 0.
- Caveat: stage 1 is not deterministic between runs, so accel can move by about ±0.05 independently of this code. iter10 does not touch tracks.

## Remaining YOLOPv2 gap (lanes; YOLO has no road edges)
- **Lane wjit:**

  | clip | iter10 | YOLO |
  |---|---|---|
  | exit | 0.017 | 0.007 |
  | late | 0.019 | 0.009 |
  | final | 0.012 | 0.008 |
  | lane_split | 0.133 | 0.092 |

  The residual is slow width drift (over seconds), not frame noise. The next step would be a per-z ground-pitch correction or a lane-pair width prior conditioned on "no lane change" (the run-mean prior already gives lane_split 0.06).
- **final lane presence 93.3/90:** its right lane is born at f5.
- **exit right-lane jit2:** 0.014 vs 0.009.
- **Ahead of YOLO:** lane presence on lane_split, jit2 L on every KITTI clip, jit2 R on final, late and lane_split, city wjit, and track accel (about 5× smoother).
