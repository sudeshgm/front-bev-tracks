# iter12 SPEC: FINAL (24 Nov 2026, IST)
iter12 is iter11 plus four changes, all in the main road-edge chain (`lines_mt.py`):
1. **Hold-mode clip-start edge backfill** (`FBEV_EDGE_BACKFILL_MODE=hold`): fixes the city/late edge wjit regression.
2. **Backfill source must cover z_ref, plus a near fill and a diagonal-copy gate** (`FBEV_EDGE_BACKFILL_COVER=1`): frame-0 edges.
3. **1.0 s edge backfill window** (`FBEV_EDGE_BACKFILL_S=1.0`, lanes stay at 0.6 s): a2d2a f0 R edge.
4. **LaneAnchor**, a lane-anchored coast of a missing main edge (`FBEV_EDGE_ANCHOR_S=3.0`, `FBEV_EDGE_ANCHOR_DEV=0.15`): munich R f561-577 and f165-166.

Plus one diagnostic dump: `edges_raw_held` (the BEV tracker coast flag of each raw main edge) in the polylines, so
replays of the iter12 GPU run are closer (see "Replay fidelity").
Detection, gating, association, tracks, lanes, the trim rules, kerb rule and ramp, EdgeHold, the fragment veto, the
gore rules, pair smoothing and lane backfill are unchanged. There is no TwinLiteNet or YOLO in the pipeline; YOLOPv2 is
only the measurement baseline. NumPy-1 compat and `FBEV_REUSE_DETS` still work.

`patch.diff` is against `iter11_spec/src` (`lines_mt.py`, `foundation_bev.py`, `run_iter11.py` → `run_iter12.py`).
Apply from a directory holding a copy of iter11's `src/`: `git apply -p2 patch.diff`.

## Replay (`tools/replay_full.py`, `tools/replay_iter12.sh`, `replay/`)
iter11's GPU polylines carry `step` (exact in-step lanes and split edges) and `edges_raw`, so the whole post-tracker
chain replays on CPU with no lane proxy and no kerb oracle: `trim_edge` → `StageLines.post_edges` (kerb, fragment veto,
EdgeHold, retrim, frag_extend, LaneAnchor, flare, DivergenceGore) → GoreTracker → lane fallback +
`suppress_edge_duplicates` → `centre_smooth_lines` → `pair_width_smooth` → `lane_backfill` → `edge_backfill` → track
linking → `smoothness_metrics.py`. `tools/replay_iter12.sh` runs it with iter11 code (`rep11`, fidelity check) and
iter12 code, then `tools/cmp_runs.py` vs the GPU run and YOLOPv2. Output: `replay/replay_vs_iter11.md|json`,
`replay/gore_rep11|iter12.txt|json`, overlays in `replay/overlays/` (top rep11, bottom iter12; lanes green, edges
orange, anchored magenta, backfill cyan, held thin).

### Replay fidelity (rep11 = iter11 code vs the iter11 GPU output)
- Gores, accel, idsw, every lane number and every edge number on the 8 non-munich clips match the GPU.
- munich R z10: 786 vs GPU 777 (+f144-155, −f615); held frames 89 vs 221. Cause: the BEV tracker coast flag of the raw
  edges was not dumped by iter11, so the replay treats coasted raw edges as live. iter12 dumps it (`edges_raw_held`)
  and `replay_full.py --held auto` uses it. On the iter12 CPU smoke this narrows but does not close the gap (late R jit2
  smoke 0.026 / auto 0.025 / off 0.022; munich R 635 / 644 / 646), so treat replay munich numbers as ±10 frames.
- Because of this, the replay treats tracker-coasted edges as live. LaneAnchor was made to do the same live (see change 4),
  so the replay is a faithful prediction for the iter12 code paths.

### Results: GPU iter11 → replay iter12 (YOLOPv2 where it has a number; YOLO has no road edges)
| clip | lane wjit gpu11 → iter12 (YOLO) | edge wjit gpu11 → iter12 | edge wjit live-only | edge pres L/R gpu11 → iter12 | f0 laneL,R edgeL,R gpu11 → iter12 |
|---|---|---|---|---|---|
| sample | 0.001 → 0.002 (0.006) | 0.022 → **0.020** | 0.019 / 0.021 | 100/100 = | ✓✓✓✓ = |
| exit | 0.007 = (0.007) | 0.047 → **0.042** | 0.048 = | 100/100 = | ✓✓✓✓ = |
| late | 0.006 = (0.009) | 0.041 → **0.034** | 0.035 = | 100/100 = | ✓✓✓✓ = |
| final | 0.006 = (0.008) | 0.026 → **0.022** | 0.023 = | 100/100 = | ✓✓✓✓ = |
| city | 0.001 = (0.005) | 0.076 → **0.037** | 0.054 = | 100/100 = | ✓✓✓✓ = |
| lane_split | 0.131 = (0.092) | 0.115 = | 0.127 = | 100/100 = | ✓✗✓✓ = |
| munich | 0.006 = (—) | 0.041 → 0.040 | 0.041 → 0.039 | 100/86.3 → 100/**96.8** | ✓✓✓✗ = |
| a2d2a | — | 0.065 → 0.061 | 0.065 = | 97.7/94.8 → 97.7/**98.8** | ✗✓✓✗ → ✗✓✓**✓** |
| a2d2b | — | 0.057 → 0.056 | 0.057 → 0.056 | 97.1/100 = | ✗✗✗✓ = |

(sample lane wjit 0.001 → 0.002 is the replay's own offset: rep11 is already 0.002.)
- **No regressions:** gores identical (exit f21-26, munich f760-765 only; replay vs GPU gores identical on all 9
  clips), accel and idsw identical on every clip, no lane or edge presence drops anywhere, lane jit2 identical.
  Edge jit2 moves by ≤ 0.003 (largest: city L 0.018 → 0.021, final R 0.021 → 0.023, munich R 0.021 → 0.023).
- **munich R z10: 777 (GPU) → 871/900 (96.8 %)**: gained f13, f144-155 (replay offset), **f165-166**, f189-251,
  f535-541, **f561-571, f576-577**; lost f615 (replay offset, also lost by rep11). Expected on GPU: about 777 + 85.
  Still missing: f572-575 (the live piece is a diagonal hook 2 m off the anchor, correctly not joined), f0 (see below).
- Kerb frames: munich 340 (rep11 342), lane_split 15, a2d2a 63. Held frames: munich 168 (replay).

## 1. City/late edge wjit regression: root cause and fix
- iter11's clip-start edge backfill extrapolated the first live edge backwards with its **constant velocity** (CV).
  City's L edge migrates from the parked-car bottoms onto the kerb at 0.18 m/frame in the first second. CV continued
  that ramp backwards, so the backfilled frames carried a steep width ramp and the median |Δ half-width| flipped to
  the ramp: city edge wjit 0.054 (iter10) → 0.076 (iter11); late 0.035 → 0.041 the same way.
- Fix: a random-walk prior, as iter11's pair-width KF already uses. Before its first measurement the edge's best
  estimate is the first live offset: `xj += clip(x0[iz] − xj[iz], ±0.5)` (`hold` mode).
- Result: city 0.076 → **0.037**, late 0.041 → **0.034**, exit 0.047 → 0.042, final 0.026 → 0.022, sample 0.022 → 0.020.
- **Disclosure:** live-only edge wjit (backfilled/anchored nearest lines excluded) is exactly iter10/iter11 (city
  0.054, late 0.035). The gain below iter10 comes from constant backfilled frames, not from better live edges.
- Rejected (replays in `iter12_work`): no edge backfill (city 0.054 but late R presence 100 → 87.5, exit 95/95, city
  93.3/93.3); `travel` mode, i.e. CV off (city stays 0.076); a 0.3 s window (city 0.076, late R 87.5).

## 2+3. Frame-0 completeness
| clip | missing at f0 (GPU iter11) | iter12 | why |
|---|---|---|---|
| a2d2a | R edge, L lane | **R edge present** (wall kerb, overlay `a2d2a_f0000.jpg`) | the R edge is born at f8 (0.8 s), past the 0.6 s window; window 1.0 s for edges |
| munich | R edge | still missing (correct) | main_R starts at z 17 m until f8; the cover rule picks the first record reaching z 10 (f15), but its 14-frame copy is kinked and crosses the lane, so the diagonal-copy gate rejects it. The real kerb is behind the stopped queue / at the image border (overlay `munich_f0000.jpg`) |
| lane_split | R lane | still missing | the only candidate is the parking-bay line id 33, 0.5 m inside the R edge, seen 3 of 4 frames, unconfirmed; YOLOPv2 has no R lane there either |
| a2d2b | both lanes, L edge | still missing | no lane reaches z 10 in the first second; the L edge copy is diagonal (gate rejects it) |
| a2d2a | L lane | still missing | no lane reaches z 10 in the first second |
- `EDGE_BACKFILL_COVER=1`: the source is the first live edge within the window that covers z_ref (held records skipped
  once an earlier record exists); earlier partial records of the role get a near fill (samples added only below their
  near end, only if the copy meets them within 0.6 m, must pass `trim_edge` without losing samples); a copy whose near
  part (z ≤ 20) deviates in slope from the lane reference by more than `EDGE_FRAG_DEV` (0.25) is rejected.
- The CPU smoke shows that the gate matters: with classical masks iter11 backfilled exit f0 R with a hook across the lane and a2d2a f0 with an X
  of diagonals; iter12 leaves them empty (`cpu_smoke/evidence/`).

## 4. Munich R f561-577 and f165-166: LaneAnchor
- Root cause: on f561-577 the raw main_R runs along a moving car's bottom and its far piece is a forecourt hook. Trim and
  the fragment veto (correctly) leave nothing, and the 0.5 s EdgeHold either expires or carries the hook outward by 0.25
  m/frame. iter10 drew the zigzag; iter11 drew nothing. f165-166: the only line is an EdgeHold-coasted kerb line from z 22 m.
- `LaneAnchor` learns, per role, the offset at z 10 between a live main edge (not held/backfill/anchored; kerb-pushed
  allowed) that runs parallel (|slope diff| ≤ 0.15 over z ≤ 25) and (a) the nearest same-side lane (stored if ≥ 0.5 m
  outside it), (b) the opposite live main edge (stored if ≥ 2.5 m). For up to 3.0 s after the last learn, a **missing or
  EdgeHold-coasted** edge (`hold_age` set) is drawn as reference + offset (a first, b if no same-side lane), if `trim_edge`
  accepts it and it reaches z 10; marked `held`, `anchored`. A held kerb line that still reaches z 10 is kept. A live
  edge that starts beyond z 10 gets anchored samples below its near end, shifted to join, if the gap is ≤ 0.6 m.
- **Edges coasted by the BEV tracker (held, no `hold_age`) are treated as live and never replaced.** The first version
  replaced them too; the GPU replay cannot see that (iter11 did not dump the flag) but the CPU smoke did: edge jit2
  lane_split L 0.025 → 0.081, exit L 0.016 → 0.029, late R 0.022 → 0.035. Fixed; the GPU replay is unchanged by the fix.
- Overlays: f565 anchored line on the kerb; f240 plausible; **f200 runs along the parked-car bottoms (questionable,
  please check the GPU still)**; f165 at the image border.
- Rejected: 1.5 s anchor (last good learn ~f485, never fires); lane-only reference (no lane on some frames);
  `anc0`/`anc15` variants in `iter12_work`.

## Lane_split lane wjit (0.131 vs YOLO 0.092): not changed, structural
- The nearest-pair identity switches through the split children (R 33 → 63 → 76 → 83; L 1 → 59 → 34); every switch is a
  real width step. L lane 34 is in fact the left kerb, kept by the nearest-lane exemption in `suppress_edge_duplicates`;
  removing it drops L presence by more than 5 pts.
- On YOLOPv2's 6 common steps (f23-29) it is YOLO 0.092 vs ours 0.186; YOLO's own jit2 there is 0.257/0.721 (ours 0.011/0.023).
- Rejected: `FBEV_PAIR_KF_MIN_S=0.3` (wjit 0.082 but jit2 0.011/0.023 → 0.050/0.114, and late R presence 97.5);
  `FBEV_KERB_TRAFFIC=under` (kerb push when a long line runs under the car: city L to −9.8 m, city L jit2 0.019 → 0.065,
  lane_split edge wjit 0.130, munich kerb frames 342 → 426). Kept as an option, default `outer` = iter9-11.

## Tests and smokes
- `tests/`: **67 passed**, normal and `PYTHONPATH=tests/np1_sim` (NumPy-1 sim). New `test_iter12.py` (15 tests): hold-mode
  backfill, cover/near fill, diagonal-copy gate, 1.0 s window, LaneAnchor (fill after hold expiry, follows the lane not
  the travel, no learning from diagonals, opposite-edge fallback, kerb-line keep, tracker-coast keep, near-end join only
  on a join), anchor off = iter11, defaults.
- **CPU smoke** (`cpu_smoke/`, all 9 clips, iter11 GPU dets + classical masks, iter11 code vs iter12 code,
  `cpu_smoke/cpu_smoke_iter11_vs_iter12.md`): 18/18 rc 0. **accel, idsw, gores, all lane numbers identical.** Edge
  presence up on city (93.3/76.7 → 100/100) and a2d2b (94.2/97.7 → 100/99.4), munich R +9 frames; edge wjit late
  0.053 → 0.038, a2d2b 0.085 → 0.079. Down: exit R 100 → 95.0 and a2d2a L 82.6 → 80.2, both = frames whose iter11
  backfill was a lane-crossing hook / diagonal X (overlays in `cpu_smoke/evidence/`), within the 5 pt rule.
  Largest edge jit2 rise: late R 0.021 → 0.026.
- **Kernel smoke** (`FBEV_SMOKE=1 FBEV_MAXF=8`, NumPy-1 sim, `src/run_iter12.py` on `smoke_kernel/ds`): RC 0, fails none,
  all 9 clips version iter12, `edges_raw_held` in the polylines (`replay/kernel_smoke_tail.txt`).

## GPU Bot run
- Entry: `src/run_iter12.py`. Dataset = this folder: `src/`, `cams_iter12.json` (identical to iter11), `yolo_baseline/`,
  plus the same clip mp4s, cams.json and motion_assumptions.json as iter11.
- Clips: all 9 (leave `FBEV_TAGS` unset). **Full stage 1+2** (no `FBEV_REUSE_DETS`). **Defaults only.**
- Env vars: `FBEV_OUT`, `FBEV_TMP`, `FBEV_DS` as before; `FBEV_YOLOPV2` + `FBEV_YOLO_REPO` (YOLO baseline on, as before);
  `FBEV_EXTRA_STILLS` = `0,1,2,3,4,5,6,7,8,9,10,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,34,60,63,80,86,100,107,119,121,123,133,140,146,160,165,166,170,180,189,200,220,240,251,258,266,267,300,450,520,523,526,535,538,541,545,557,560,561,565,568,570,572,574,577,587,603,615,625,647,677,692,697,700,760,762,878,879,880,881,882,883,884,885,886`
  (the iter11 set plus f9-15 (a2d2a R edge birth, munich f0-15 backfill), munich 165/166 (anchored), 189/200/220/251
  (anchored run, f200 questionable), 535/538/541, 561/568/570/577 (anchored), 572/574 (still missing), 615).
- Do not set `FBEV_SMOKE`, `FBEV_MAXF`, `FBEV_REUSE_DETS`, `FBEV_WORLD_PIN`, `FBEV_LINE_SMOOTH_POSE`, `FBEV_TRIM_DEBUG`,
  `FBEV_RAY_R`, `FBEV_REPIN`, `FBEV_ROBUST_NIS` or any of the new iter12 knobs.
- New defaults (no need to set): `FBEV_EDGE_BACKFILL_MODE=hold`, `FBEV_EDGE_BACKFILL_COVER=1`, `FBEV_EDGE_BACKFILL_S=1.0`,
  `FBEV_EDGE_ANCHOR_S=3.0`, `FBEV_EDGE_ANCHOR_DEV=0.15`, `FBEV_KERB_TRAFFIC=outer`. iter11 defaults unchanged.
- Outputs to `/workspace/foundation_bev/iter12/`: as iter11 (`results/<clip>_foundation_bev_iter12.mp4`, stills, tracks,
  polylines with `step`, `edges_raw` and the new `edges_raw_held`, dets_dump, dets, metrics.md, ego_speed_table.md,
  metrics_yolo_vs_run.md).
- Expected T4 time: ~75-80 min (as iter11; the new passes are CPU-side and cost seconds).
- **Please produce `metrics_vs_iter11.md`:** 9-clip table iter11 (`/workspace/foundation_bev/iter11`) → iter12 with
  accel_rms, n_tracks, idsw, lanes L10/R10 pres + jit2 + wjit, edges L10/R10 pres + jit2 + wjit; gore frames per clip;
  held / kerb_occ / frag / backfilled / anchored edge frames per clip; munich R z10 frame count and the frames gained or
  lost vs iter11; frame 0 lanes/edges on every clip; flags for any presence drop > 5 pts, accel rise > 0.05, idsw > 0.
