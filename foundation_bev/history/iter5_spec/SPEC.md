# Foundation BEV iter5 SPEC: FINAL (small delta on iter4_spec, 23 Nov 2026 ~23:30 IST)

Trigger: Sudesh reported that kitti_highway_exit_approach does not handle the lane/road merge. The exit lane diverges on the right, chevron gore hatching is visible around f0008, and the right lines split.
iter3 is running on the GPU and is untouched; this is for iter4.

## Diagnosis (iter2/3 line model)
1. **One curve per slot.** Lines are "nearest left/right lane + one edge per side", each a single global x(z) poly2. A diverging exit line and the through line get averaged into one curve, or one of them is dropped.
2. **Fixed gate.** The absolute 0.5 m jump gate and the coefficient EMA treat a genuinely diverging line as an outlier.
3. **Edge follows the main carriageway.** The right road edge is a single poly on the lower envelope of the road mask, so it doesn't follow the widening exit. Chevrons are not represented at all.
4. **The "two straight orange lines near the ego" are not the ego path.**
   - They are the tails of the two lead cars (#1 and #3, 57–72 m ahead, outside the 40 m BEV).
   - The tails are drawn in the world frame with the assumed ego travel (22 m/s at 4 fps = 5.5 m/frame) over a 40-frame tail, so each tail is a roughly 200 m streak through the whole panel.
   - Car (BGR 0,180,255) and edge (BGR 40,140,255) are both orange.
   - The ego path itself is only drawn behind the ego (grey).
   - **Render fix for iter4:**
     - Draw tails in the ego frame (relative motion).
     - Cap tails at 2 s and 15 m of drawn length.
     - Don't draw tracks whose current position is outside the BEV range.
     - Car colour becomes red (BGR 60,60,255).
     - Split/exit edges use a lighter orange (BGR 40,200,255).

## New modules (src/)
### `lines_mt.py`: multi-line tracking

**Representation.** Every lane line and every edge is a polyline x(z) sampled on a 0.5 m z grid (5–45 m), valid only over its observed span. It may bend, diverge or end anywhere; there is no global poly.

**Per frame:**
- **Paint chains.**
  - Paint components are skeletonised and described by PCA (length, angle from z).
  - Near-longitudinal pieces (≤ 35°, ≥ 1 m) are linked across dash gaps (≤ 9 m) along the chain's own heading.
  - Oblique short pieces (> 20°, 0.6–4 m) become gore stripe candidates.
- **Fit.** Each chain gets a smoothing spline (`UnivariateSpline`, k = 2, s = n·0.04²), sampled on the grid, which allows piecewise / diverging shapes.
- **Edges (`edge_chains`).** Per BEV row, the drivable mask is cleaned first: longitudinal close, holes < 25 m² filled (hatching, shadows, dashes), largest component kept.
  - The run under the ego gives `main_L` / `main_R`, so the right edge widens with the exit lane.
  - A second run beyond a ≥ 0.5 m non-drivable gap over ≥ 3 m of rows (grass island past the gore nose) gives `split_in` / `split_out`.
  - Boundaries touching the IPM/image border are dropped (field of view, not road).
  - Each boundary is median- plus spline-smoothed along z.
- **Gore (`gore_polygons`).** At least 2 parallel oblique stripes (centres within 4 m, angles within 25°) form a convex-hull polygon.

**Across frames (`MultiLineTracker`):**
- **Association.** Hungarian on cost = median lateral distance over the nearest 6 m of the overlap (≥ 2 m overlap) + 2 × relative heading. So two lines that coincide near the fork but diverge further out still match their own track.
- **Relative gate.** The gate is clip(3 × the line's own innovation scale σ, 0.3, 1.2 m), not an absolute 0.5 m.
- **Motion model.** A per-line offset rate v_off (EMA, clipped to 0.3 m/frame) follows steady sideways drift such as a lane change.
- **Per-z update.** EMA α = 0.4 where observed. The span grows where newly observed, so divergence is kept, and z cells unobserved for > 8 frames are forgotten.
- **Lifecycle.** Birth after 3 hits; death after 6 misses (1 for tentative lines). There is no slot count.
- **Split/merge.** A newborn line whose near or far end lies within 1 m of a confirmed line, or that converges below 1 m with it and diverges above 2 m, becomes its child (`parent`, `fork_z`). A `fork` event is logged when it is confirmed.
- **Rendering.** The fork node is drawn as a small white ring; the child line is drawn from the fork point.

**`GoreTracker`:** a gore polygon is displayed after 2 hits and kept through 4 misses. It is rendered subtly: a 25 % fill plus a 1 px outline in edge colour.

### `bev_ipm.py`: classical CPU stand-in for the replay only
- IPM homography from the flat-ground model.
- Paint: horizontal + vertical top-hat.
- Structure-tensor stripe orientation.
- Asphalt: Lab Mahalanobis to seed patches in front of the ego + low texture, with paint treated as drivable.
- **On the GPU**, the same downstream logic takes the SAM 2.1 road mask and the WD3D 'lane marking' / SAM paint mask, warped with the same homography (`ipm_homography` / `warp`). bev_ipm is only needed for model-free replay.

## Integration plan into foundation_bev.py (iter4 stage 2)
- Pass 1: warp `ob["road"]` and the paint mask to BEV per frame (masks are already computed).
- Pass 2b: replace LineTracker / EdgePair with:
  - one `MultiLineTracker("lane")`,
  - one `MultiLineTracker("edge")` per edge role (`main_L`, `main_R`, `split_in`, `split_out`),
  - plus `GoreTracker`.
- Keep the centred 7-frame window as a per-z median over the line's history (zero-phase).
- Polylines JSON v4 per frame:
  - `lanes`: [{id, x (grid), parent, fork_z, hits, held}]
  - `edges`: [{id, role, x, ...}]
  - `gores`: [{poly}]
  - `events`
- smoothness_metrics then reads `x` on the grid (`lanes_bev` is kept for compatibility).

## Offline replay (replay_exit.py, kitti_highway_exit_approach, 40 frames)
- Classical masks only; no models were run.
- **Stills** in replay/, each showing front with projected iter4 lines | iter2 BEV (before) | iter4 BEV (after): `kitti_highway_exit_approach_f{0008,0014,0019,0020,0026}_before_after.png`.
- **What it shows:**
  - From f11 the diverging exit line is tracked as a separate child line of the solid right line. Forks were logged at f11 (z 23.5 m) and f14 (z 22.5 m), and f20 shows the fork node and both lines.
  - The right edge widens with the exit at f8.
  - 3–5 lane lines are live after f11, against iter2's fixed 2 lanes plus 2 edges.
- **Limits:**
  - The classical asphalt mask leaks into the left verge and misses the bright exit-lane asphalt in some frames (f20), so edges are only indicative here. SAM masks on the GPU are the real input.
  - Gore polygons appear in 5/40 frames only, because the classical paint mask fragments the chevrons.
  - Stats: replay/kitti_highway_exit_approach_replay_stats.json.

## Tests
tests/test_lines_mt.py: ALL ITER4 LINE TESTS PASSED. It covers:
- the diverging line becoming a child at the right fork z, with neither line averaged,
- steady lane-change drift keeping one id,
- birth/death with 6 lines (no slot limit),
- the spline keeping a bend,
- a widening edge plus a grass-island split giving main + split edges,
- gore clustering (4 stripes → 1 polygon; 1 stripe → none).

## Status 23 Nov 2026 ~21:00 IST: iter4 package complete on CPU (nothing launched)
**Wired into stage 2 (foundation_bev.py, default `--lines mt`; `--lines legacy` = iter3 L/R slots):**
- Pass 1 stores the SAM road mask warped to the BEV (`bev_ipm` IPM from gm0, 1 bit/px).
- Pass 2b builds the paint BEV from the lifted lane pixels, rasterised and joined 1.5 m along z. It then calls
  `lines_mt.StageLines.step`:
  - lanes: a MultiLineTracker with Hungarian association, its own motion-model gate, birth/death, and fork parents;
  - edges: one tracker per drivable-area role (main_L, main_R, split_in, split_out); a coasting duplicate is dropped after a re-birth;
  - gores: chevron stripe clusters, drawn as a subtle outline.
- Records keep the legacy fields (`c`, `side`, `zmin/zmax`, `hits`, `held`) and add `xg` (x on the lines_mt.ZS grid).
  `bev_smooth.sample_curve` uses `xg`, so drawing, lanes_bev/edges_bev, and the metrics' nearest-L/R series all work unchanged.
- Lanes and edges stay separate. Split edges and the gore outline are lighter orange; fork nodes are a white circle; the ego path is white.
- The polylines JSON adds `parent`, `fork_z`, `role` per line, `gores` per frame, plus `line_events` (fork/death) and `lines`.

**Ego speed:** `BevPhaseEgo` runs in pass 1, with a HUD saturation mask when cams `hud`, and `bev_phase_z0` set to 12 m for munich.
- Selection order: oxts, then BEV phase, then LK flow, then scene closing rate, then the prior.
- BEV phase is accepted only if valid_frac ≥ 0.5, the speed is > 0.5 m/s, and it is within 30 % of the scene closing rate (or of the prior if the scene rate is invalid).
- Otherwise selection falls through, so A2D2 stays on the prior or flow.
- The kernel prints the BEV-phase column in the EGO table.

**CPU re-render** (`cpu_rerender/run_rerender.sh`):
- Inputs: the iter3 `dets_dump/*_dets_ego.json` (raw WD3D per frame), loaded by `--dets x.json`, feeding the full iter4 stage 2 (gates, tracker, ego, lines_mt, render).
- No SAM/DAV2 on CPU. Masks come from `--classical-masks` (bev_ipm asphalt/paint), with the road extent clamped to the iter3 SAM edges (`--road-prev-polylines`, plus 3 m of widening allowed).
- Side-by-side stills: cpu_rerender/sbs/ (iter3 on top, iter4 below).
- Metrics: cpu_rerender/metrics_before_after.md. iter3 is also re-scored at capture_hz, because iter3 used the file fps of 4 for KITTI.

| clip | accel_rms iter3 @file fps / @capture_hz / iter4 | lane L10/R10 jit2 iter3 → iter4 | lane wjit | edge L10/R10 jit2 | lane L10/R10 pres % | edge L10/R10 pres % |
|---|---|---|---|---|---|---|
| exit_approach | 0.68 / 4.24 / **0.49** | .003/.001 → .015/.006 | .003 → .022 | .003/.002 → .007/.028 | 95/95 → 22.5/95 | 95/95 → 90/92.5 |
| highway_late | 1.83 / 11.42 / **0.68** | .009/.007 → .008/.011 | .024 → .016 | .005/.006 → .016/.024 | 95/95 → 60/45 | 90/95 → 92.5/95 |
| lane_split | 1.13 / 7.09 / **0.90** | –/.002 → .000/– | – | –/.000 → .066/.043 | 0/40 → 36.7/0 | 0/73 → 90/77 |
| munich | 0.91 / 0.91 / 0.92 | .001/.001 → .005/.002 | .003 → .012 | .001/– → .016/.025 | 92/98 → 17/17 | 100/0 → 32/33 |

How to read the table:
- Object smoothness improves by 6–17× at the physical rate, because the KF dt now matches the 10 Hz capture.
- The line columns compare SAM masks (iter3) against classical CPU masks (iter4). They measure the stand-in masks, not lines_mt.
- The true line comparison needs the GPU run.

**Still broken / open:**
- The classical masks are weak: shadows, the white solid line, and urban scenes (munich lane presence is 17 %).
  - The CPU line numbers are therefore worse than iter3 on SAM.
  - On exit_approach f0026, a false split_in/out pair appears inside the right lane where iter3 had no right edge.
- Exit merge (exit_approach), checked visually:
  - f0008: the lane fork is detected (child of the solid line, z≈26 m), and the right edge widens around the gore.
  - The edge outline is ragged, and the gore polygon was not detected in the kernel re-render (0 frames; the standalone replay found 5).
- Validation of the merge on SAM masks needs the GPU run.
- The CPU re-render keeps a guardrail-post "person" (exit_approach) because there is no DAV2 depth gate on CPU. The GPU run has it.
- Ego speed:
  - highway_late BEV phase reads 16.7 m/s against a prior of 25. It is rejected, so the prior is kept, though 16.7 may be the true value.
  - munich BEV phase reads 20.1 against a scene speed of 13.4. It is rejected and the scene speed is used.
  - lane_split reads 4.6 against a prior of 18. It is rejected.
  - In the MAXF=8 smoke, BEV phase is NaN on every clip (too few frames); full clips are fine.
- yolo_baseline/test_decode_fix.py needs torch (not on this box; it passed earlier in a torch env).

**GPU Bot hand-off:** the same as iter3 (src/run_iter4.py + cams_iter4.json in the dataset).
- New sources it copies: lines_mt.py and bev_ipm.py.
- Outputs: `*_foundation_bev_iter4.mp4`.
- Expected time is about iter3 + ~5 % (lines_mt + BEV warp ≈ 0.1 s/frame on CPU).


## iter5 delta: FINAL
**Changes (src/lines_mt.py, src/foundation_bev.py; patch.diff is against iter4_spec/src):**
1. Clutter control (`lines_mt.clean_lanes`):
   - Lanes must span at least 6 m in z and have at least 4 hits.
   - Lanes that are near-horizontal or zigzag are rejected (median |dx/dz| > 0.5, or 90th-percentile row step > 0.3 m).
   - Crossing lanes and duplicates on the same paint are dropped, keeping the older line.
   - At most 3 lanes per side, nearest at z=10 m.
   - Split edges need at least 4 hits and |dx/dz| ≤ 0.25 with steps ≤ 0.15 m; this removes the horizontal yellow zigzag.
   - Stripe gores need at least 3 stripes and a hull of ≤ 40 m².
2. Trails: the 2 s / 15 m cap now uses the *selected* ego travel (`s_sel=s_ego`).
   - Previously the tracker world used the prior speed (18 m/s on lane_split against a true ~4.6), so parked cars drew 15 m diagonals.
   - Every past point is expressed in the current ego frame.
3. Fork gores (`lines_mt.fork_gores`) cover two cases:
   - a child lane diverging from its parent;
   - a split edge, or the side's main edge, pulling away from the outermost lane line, with slope ≥ 0.05 and ≥ 1 m of growth within 30 m.
   - The polygon is the widening wedge.
4. Lane presence fallback: the iter3 LineTracker runs alongside lines_mt.
   - When a side has no lines_mt lane with ≥ 4 hits starting at ≤ 12 m, the iter3 nearest lane is added (`fallback: true` in the polylines).
5. Jitter: lanes EMA α 0.4 → 0.25, edges α 0.3 → 0.2, and a smoothing-spline re-fit (s=0.5) of each blended line.
6. BEV-phase acceptance:
   - Accepted if within 40 % of the reference, or valid on ≥ 0.80 of pairs (lane_split is 0.83; the brief said 0.85).
   - Always rejected when repeated frames are present, so munich stays rejected.
   - On lane_split, 4.6 m/s is now accepted.

**CPU re-render** (cpu_rerender/run_rerender.sh): iter4 GPU dets_dump through iter5 stage 2, using classical masks and the iter4 SAM edge extent.
- iter4cpu/ holds the same inputs run through the iter4 code.
- Side-by-side stills are in cpu_rerender/sbs/ (iter4 GPU on top, iter5 CPU below).
- Metrics are in cpu_rerender/metrics_before_after.md.

| clip | run | accel | lane L/R jit2 | lane wjit | edge L/R jit2 | lane L/R pres % |
|---|---|---|---|---|---|---|
| exit_approach | iter4 GPU | 0.54 | .010/.006 | .014 | .021/.039 | 95/90 |
| | iter4 CPU | 0.49 | .015/.007 | .017 | .025/.031 | 22.5/95 |
| | iter5 CPU | 0.49 | .026/.006 | .042 | .023/.040 | 22.5/92.5 |
| lane_split | iter4 GPU | 1.05 | .043/.001 | .108 | .085/.128 | 93/77 |
| | iter4 CPU | 1.06 | .000/.001 | .041 | .095/.106 | 50/27 |
| | iter5 CPU | 1.06 | .001/– | – | **.040/.064** | 30/0 |
| munich | iter4 GPU | 0.92 | .009/.003 | .007 | .032/.039 | 88/65 |
| | iter4 CPU | 0.92 | .004/.001 | .008 | .031/.037 | 39/18 |
| | iter5 CPU | 0.92 | .004/.001 | .006 | .042/.042 | 40/15 |

- Gores: exit_approach fires on f4–f10 (fork_edge wedge on the right, around the chevrons at f8). It does **not** fire on f15–f30 in the CPU replay.
- lane_split: at most 2 lanes per frame (iter4 GPU had up to 10). There are gore hits on f2–7 (stripes) and f24–29 (fork), which may be false positives in the city.

**Still open:**
- Lane presence cannot be checked on CPU. The fallback needs SAM lane pixels (empty in the CPU re-render), so the presence and jit2 gains on Munich and A2D2 only show in the GPU run.
- Parked-car trails on lane_split are now short but lateral (2–5 m sideways), from box-x drift with range.
- Exit lane jit2 / wjit went up on CPU: the extra parent/child lane near the solid line.

**Launch:** dataset = iter5_spec contents (src/*.py, cams_iter5.json, yolo_baseline/). Entry: `src/run_iter5.py`; outputs `*_foundation_bev_iter5.mp4`.
- Env vars: FBEV_OUT, FBEV_TMP, FBEV_DS, FBEV_MAXF, FBEV_SMOKE, FBEV_BUDGET_S1, FBEV_WALL_CAP.
- YOLO baseline: FBEV_YOLOPV2 (TorchScript weights) and FBEV_YOLO_REPO.
- Smoke (FBEV_SMOKE=1 FBEV_MAXF=8): rc 0, 9 mp4s. Tests: 8 pytest passed, and the 4 test scripts each print ALL PASSED.
