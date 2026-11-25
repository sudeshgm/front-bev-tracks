# iter15 GPU run (T4, rc 0) - review of the non-identical tracks, the wjit moves and the munich pitch

## (a) Tracks not byte-identical vs iter13 on munich / a2d2a / a2d2b: stage-1 detector wobble, not iter15
- Stage-1 WildDet3D dets (`dets/wd3d_<clip>.npz`, md5 of boxes): KITTI x6 identical in iter12 / iter13 / iter15.
  munich / a2d2a / a2d2b: **iter15 == iter12 byte-for-byte** (24116 / 4082 / 5138 boxes); iter13 is the odd one out
  (24126 / 4073 / 5128). Same weights, same code path (stride 1): GPU nondeterminism in stage 1 with two outcomes.
- iter13 vs iter15 car/person/bicycle dets differ only on a2d2a f61, f104, f145; a2d2b f64, f135; munich f282.
- **iter15 tracks CSV are byte-identical (md5) to iter12's** on munich, a2d2a, a2d2b (rows, ids, class, bev, filt, score).
  So the "new" a2d2a bicycle id 32 (f113-138, 26 rows, stationary at x -8.8 m z 14 m, median score 0.24) and the a2d2b
  person id 14 (iter13: id 13; same 12 rows f160-171, same class) are the iter12 outcome of the same detector wobble.
- munich: the only raw per-frame observation difference is f282 (= the det change; track 35 row lost); track bev
  diffs f230-337 (tracks 32, 34-39, RTS/backfill window), line diffs f206-366, ego speed identical (scene 13.13).
- The tracker never sees ego yaw / pitch: trk.step / trk.finalize run before EgoMotion.finalize and the yaw / pitch
  code; world pin is off. The 6 KITTI clips (identical dets, new yaw everywhere) have identical tracks.
- a2d2 ground model and lines follow the dets too: a2d2a ground pitch iter12 = iter15 = -0.7748 deg (iter13 -0.7665);
  lines iter12 vs iter15 differ on 3 / 13 frames (iter13 lane-code changes) vs 157 / 106 for iter13 vs iter15.
- The CPU check (iter14 vs iter15 on CPU) used the iter12 dets on both sides, hence "identical everywhere"
  (munich was not finished on CPU; /tmp was cleared).
- No regression. idsw 0 on all clips; a2d2a accel 0.499 -> 0.495.

## (b) late edge wjit .034 -> .041, final lane jit2 L .014 -> .027: real (deterministic), within metric noise
- Predicted by the CPU run before the GPU run (late edge .035 -> .041, final jit2 L .014 -> .031): the lines' motion
  compensation now uses the new BEV-phase speed (late 16.65 -> 18.43 m/s, OXTS 18.35; final 19.21 -> 20.28, OXTS 19.51).
  Clips whose speed did not change (sample, lane_split) have identical lines although their yaw changed.
- Paired bootstrap of the median (5000 draws): late edge wjit +0.007 m, 95 % CI [-0.004, +0.012], 25 frames worse /
  13 better; final lane jit2 L +0.013 m, CI [-0.003, +0.054], 14 worse / 12 better.
- Distribution: late edge width sd 0.541 -> 0.535 m, wjit mean .060 -> .062, p90 .103 -> .108; final lane width sd
  0.528 -> 0.521, jit2 L mean .245 -> .253, p90 1.126 -> 1.103. exit edge wjit .042 -> .041, final edge .022 -> .021.
- Verdict: a mm-level change of the median, not separable from noise; final's speed got worse vs OXTS (-1.3 -> +3.9 %),
  late's got better (-9.2 -> +0.4 %). Known issue 2 (lines see corrected speed in pitch-0 BEV metres).

## (c) munich lane pitch 0.945 deg (10 pairs, ratio 1.178): harmless, output-neutral
- bp (the re-run BEV-phase estimators) feeds only select_ego_speed; munich's BEV phase is rejected by n_repeat 450 > 0
  (iter5 rule), so the source stays scene 13.13 m/s with or without the pitch. munich polylines differ only in a window
  around the f282 det change (f206-366), nothing elsewhere; gores f760-765; R z10 871/900 unchanged.
- The estimate itself is not trustworthy on munich (dashcam, horizon_v pitch mode, -11.9 deg camera pitch, 10 pairs) and
  the re-pass costs a full re-decode. Follow-up (not in this commit, which is the code that was GPU-run): skip the
  re-pass when the BEV phase cannot be selected (n_repeat > 0 or oxts present), log theta as measured-not-applied.
