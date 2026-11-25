# iter15 SPEC (25 Nov 2026, CET): ego-motion fix

Base: iter14 (foundation-bev 528ae08; public output = iter13 GPU run /workspace/foundation_bev/iter13/). Entry
`src/run_iter15.py`. `patch.diff` = `git diff --no-index -M iter14_spec/src iter15_spec/src` (apply with `git apply -p2`
from the parent of an iter14 `src` tree; verified: applies cleanly and gives this src exactly). Models unchanged:
WildDet3D, SAM 2.1 hiera-small, Depth Anything V2-Small. No new or lightweight model; the fix is geometry on the
existing SAM road / lane masks and WildDet3D boxes.

## Changes (defaults; iter14 behaviour = `FBEV_EGO_YAW=legacy FBEV_EGO_PITCH=0`, verified parsed-identical)
1. **Yaw sign fix (all modes except legacy).** EgoMotion returned dpsi = -angle(R), i.e. + = LEFT, while every
   consumer (path integration x += ds sin psi, ego_to_world / world_to_ego, lines_mt._to_frame, the curved path
   ahead) assumes + = RIGHT. Every drawn ego path and exported `ego.psi` mirrored the turn since iter3 (KITTI
   lane_split: OXTS 11 deg/s right turn, drawn as a left turn). Now dpsi = +angle(R).
2. **BEV-ECC yaw (`FBEV_EGO_YAW=bev`, default; `em` = point-LK EgoMotion with the sign fix).** The iter14 road-masked
   BEV ROI (SAM road minus WildDet3D vehicle boxes, z 7-40 m, |x| <= 6 m, feathered, high-passed) of consecutive frames is
   aligned with Euclidean ECC (60 its), initialised with the phase-correlation shift; dpsi = -image angle. Repeated
   file frames (munich) share the next distinct pair evenly; failed pairs are 0; EgoMotion smoother (median-5,
   mean-5). EgoMotion's point-LK picks the ~0 m/s hypothesis at night / on smooth roads (fc1 median flow 0.2-0.4 m/s)
   and its rotation is noise; speed hypotheses anchored at the clip speed do not fix it (3/59 pairs on fc1_split).
3. **Lane-parallelism pitch for the BEV-phase speed (`FBEV_EGO_PITCH=1`, default).** Single-frame lane fits (the iter3
   `fit_frame_lines` path on the lifted SAM lane pixels, no aggregation, so speed-independent) give adjacent-lane width
   ratios w(28 m) / w(10 m); a pitch error theta gives r = (1 + theta 28/h) / (1 + theta 10/h). With >= 5 pairs and
   |theta| >= 0.05 deg (clamp 1 deg), the two BEV-phase estimators (bpe, bpe_rm) are re-run at p0 + theta in a re-decode
   pass (stored packed keep masks; no model calls; ~10 ms / frame); `select_ego_speed` is unchanged. Only the speed
   uses theta; the ground model of lines / tracks is unchanged. Results json: `ego_speed.ego_pitch`, `ego_speed.yaw_source`.
4. `topo_log.py` (`FBEV_TOPO_LOG`, diagnostics only, default off; output-neutral, verified on 9 public + fc1).
   The iter15 far-field cue and the other fork cues were not adopted and are not in src (offline scripts only).

## Validation references
- **KITTI OXTS (public GT, new):** every KITTI clip was matched to KITTI raw (thumbnail NCC 1.000 at the start frame;
  `tools/kitti_gt/`, `kitti_gt_map.json`, `kitti_oxts_subset.json`): highway sample / exit / late / final =
  2011_10_03 drive 0042 f300 / f719 / f1100 / f1140; city_cyclist = 2011_09_26_0001 f40; lane_split = 2011_09_26_0005 f100.
- A2D2: dashboard renders, no bus signals; munich: dashcam. No GT.
- fc1 (private, local): owner VO (reference, not GT) + the label trajectory in eval_gt.py. Aggregate numbers only.

## Ego error, before (iter14) -> after (iter15), real models on CPU (`tools/ego_eval.py`)
| clip (ref) | speed err | yaw MAE deg/s (corr) | heading drift deg | path end err m (lateral) | yaw jerk deg/s^2 |
|---|---|---|---|---|---|
| kitti sample (OXTS) | -2.7 -> -2.7 % | 0.51 -> 0.12 (-0.07 -> +0.68) | +1.6 -> +0.2 | 3.27 (1.84) -> 2.69 (0.05) | 4.1 -> 0.7 |
| kitti exit (OXTS) | -3.4 -> +4.6 % | 1.40 -> 0.16 (-0.36 -> +0.34) | -1.5 -> -0.3 | 3.34 (0.11) -> 4.37 (0.18) | 5.3 -> 0.8 |
| kitti late (OXTS) | -9.2 -> +0.6 % | 3.48 -> 0.90 (-0.78 -> +0.66) | -1.9 -> +0.7 | 6.06 (1.91) -> 1.44 (0.85) | 13.5 -> 6.9 |
| kitti final (OXTS) | -1.3 -> +3.9 % | 0.59 -> 0.20 (-0.35 -> +0.86) | -0.8 -> +0.5 | 1.27 (0.93) -> 2.07 (0.17) | 2.5 -> 1.2 |
| kitti cyclist (OXTS) | +10.7 -> +6.1 % | 0.88 -> 0.41 (-0.72 -> +0.76) | +0.1 -> -0.8 | 3.40 (0.42) -> 2.01 (0.18) | 2.3 -> 1.5 |
| kitti lane_split (OXTS) | -18.4 -> -18.4 % | 24.4 -> 1.12 (sign) | -71.3 -> +3.2 | 9.83 (9.45) -> 2.77 (0.60) | 11.1 -> 9.7 |
| fc1_split (VO) | +13.2 -> +1.0 % | 1.58 -> 1.08 (0.33 -> 0.51) | +19.9 -> -2.0 | 104 (95) -> 19.7 (19.2) / 345 m | 4.7 -> 0.8 |
| fc1_merge (VO) | +0.4 -> +0.4 % | 0.61 -> 0.28 (-0.57 -> +0.86) | +6.2 -> +0.1 | 13.0 (12.7) -> 2.3 (0.5) / 102 m | 0.7 -> 0.8 |
- Yaw better on 8/8, path end error better on 6/8 (exit +1.0 m, final +0.8 m from the speed change). KITTI mean
  |speed err| 7.6 -> 6.0 %; speed better on 3 (late, cyclist, fc1_split), worse on 2 (exit, final), same on 3.
- Sign fix alone (iter14 psi negated): lane_split yaw MAE 24.4 -> 2.3, fc1_merge corr -0.57 -> +0.57.
- fc1 eval_gt (label trajectory): split ego distance +15.5 -> +3.1 %, speed |err| 3.09 -> 0.34 m/s, position err
  median / final 37.5 / 105 -> 11.5 / 20.9 m, heading final 13.1 -> 1.7 deg; merge position 5.7 / 12.9 -> 1.3 / 3.5 m,
  heading 2.5 / 4.2 -> 0.45 / 0.11 deg. Lines: merge identical; split +-1-3 cm; gores unchanged (0/6, 0/26, 0 false).

## Public output change (CPU real models, /tmp/e15/new vs iter14 pub_i14)
- Tracks byte-identical on every clip (accel, idsw unchanged). Ego pose (drawn path, exported `ego`) changes everywhere.
- Lanes / edges change only where the speed changed: exit 23.6 -> 25.6, late 16.7 -> 18.5, final 19.3 -> 20.3, cyclist
  11.3 -> 10.8 m/s. wjit (`tools/cmp_runs.py`): exit edge .042 -> .040 (live .048 -> .045), late edge .035 -> .041
  (live .038 -> .041), final lane .011 -> .010 (jit2 L .014 -> .031), city equal; lane wjit equal elsewhere.
- sample, lane_split, a2d2a, a2d2b: lines identical (no speed change). munich: not finished on CPU (/tmp cleared); on GPU
  its ego speed is unchanged (scene 13.13), lane pitch measured 0.945 deg but unused (BEV phase rejected, repeated frames).
- Gores: exit 21-26 only (unchanged).

## Tests and smokes
- `tests/`: **104 passed** (98 iter14 + 6 `test_iter15.py`), normal and NumPy-1 sim (`PYTHONPATH=tests/np1_sim`).
- Kernel smoke (`FBEV_SMOKE=1 FBEV_MAXF=8`, NumPy-1 sim, 9 public clips): RC 0, fails none.
- Legacy identity: `FBEV_EGO_YAW=legacy FBEV_EGO_PITCH=0` = iter14 on lane_split / cyclist (polylines parsed-equal, tracks byte-identical).

## GPU run (T4, 25 Nov 2026, rc 0; brief `GPU_BOT_BRIEF.md`)
- 9 public clips, defaults. Yaw MAE vs OXTS lower on all 6 KITTI clips (sample .51 -> .11, exit 1.40 -> .13, late 3.48 ->
  .86, final .59 -> .21, cyclist .88 -> .41, lane_split 24.4 -> 1.13 deg/s); speeds within 2.1 % of the CPU expectation.
  KITTI mean |speed err| vs OXTS 7.7 -> 5.6 % (only final worse, -1.5 -> +4.0 %); path end error lower on 5 / 6 (final
  1.35 -> 2.12 m); yaw jerk lower on 6 / 6 (`results/iter15/ego_eval_oxts.txt`).
- Tracks byte-identical vs iter13 on the 6 KITTI clips. munich / a2d2a / a2d2b differ from iter13 but are byte-identical
  to iter12: stage-1 WildDet3D GPU wobble (iter15 dets == iter12 dets; iter13 differed), not iter15 (`results/iter15/gpu_check.md`).
- Gores: exit f21-26, munich f760-765 only. munich R z10 871/900 (unchanged). Night gate on 0, car audit 0 on all clips.
- late edge wjit .034 -> .041, final lane jit2 L .014 -> .027: deterministic speed effect, inside bootstrap noise.
