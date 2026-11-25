# GPU Bot brief: foundation BEV iter15 ego-motion fix (9 PUBLIC clips only) — PREPARED, NOT SENT

**No private data in this run.** Do not copy, mount or upload any fc1 clip, frame, label, odometry, cams_private.json or
motion_private.json (not to FBEV_DS, Modal, Drive or git). The run uses only the 9 public clips.

- **Code:** `iter15_spec/src/` (entry `src/run_iter15.py`) = iter14 + `iter15_spec/patch.diff`. Check the files against
  `iter15_spec/SHA256SUMS_src.txt` before the run (12 files, incl. the new `lane_pitch.py` and `topo_log.py`).
- **Dataset (FBEV_DS):** the same 9 public mp4s, `cams.json` and `motion_assumptions.json` as the iter13 run, plus
  `iter15_spec/src/`, `iter15_spec/cams_iter13.json` (cams unchanged) and `iter15_spec/yolo_baseline/` (= iter14).
- **Clips:** `FBEV_TAGS=kitti_highway_sample,kitti_highway_exit_approach,kitti_highway_late,kitti_highway_final,kitti_city_cyclist_candidate,kitti_lane_split_candidate,munich,a2d2a,a2d2b`
- **Run:** full stage 1 + 2 (no `FBEV_REUSE_DETS`). **Defaults only** (`FBEV_EGO_YAW=bev`, `FBEV_EGO_PITCH=1` are the
  defaults; do not set any FBEV_* knob).
  - YOLOPv2 baseline on (`FBEV_YOLOPV2` / `FBEV_YOLO_REPO` as in iter13).
  - Same GPU, weights and checkpoints as iter13: WildDet3D, SAM 2.1 hiera-small, Depth Anything V2-Small.
  - `FBEV_OUT=/workspace/foundation_bev/iter15/`. The night-gate log goes to `$FBEV_OUT/gate.jsonl` (diagnostics).
- **Stills (same list as iter13 / iter14):** `FBEV_EXTRA_STILLS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,63,80,86,100,107,119,121,123,133,140,146,148,149,150,151,152,153,154,155,160,165,166,170,180,189,190,191,192,193,194,195,196,200,220,240,242,250,251,258,266,267,300,450,520,523,526,535,538,541,545,546,547,548,549,557,560,561,562,563,565,568,570,571,572,574,577,587,603,615,625,647,677,692,697,700,760,762,878,879,880,881,882,883,884,885,886`
- **Expected T4 time:** about 70 min (as iter14; the lane-pitch re-decode adds ~10 ms / frame on the 5 clips with lanes).
- **After the run:** `bash iter15_spec/tools/metrics_vs_iter13_e15.sh` (OLD=/workspace/foundation_bev/iter13,
  NEW=/workspace/foundation_bev/iter15). It writes `metrics_vs_iter13.md` with:
  1. gpu13 vs gpu15 table (accel, idsw, presence, wjit, jit2, gores, f0 completeness, munich R z10 diff);
  2. night-gate on counts per clip (**must be 0**);
  3. car audit (synthetic edges inside cars must stay 0);
  4. change scope per clip (tracks / ego pose / lines);
  5. **ego vs KITTI OXTS** (`kitti_oxts_subset.json`, public KITTI raw): before / sign-fix-only / after;
  6. ego sources, yaw source and lane pitch per clip.
- **Acceptance vs iter13 (CPU-verified):**
  - every clip rc 0; gores exit f21-26 and munich f760-765 only;
  - tracks: accel / idsw equal to iter13 up to GPU noise (CPU: tracks byte-identical);
  - lines: equal to iter13 up to GPU noise on sample, lane_split, munich, a2d2a, a2d2b; on exit / late / final / cyclist
    within +-0.01 m wjit of iter13 (CPU: .042/.040, .035/.041, .011/.010, .054/.054 edge or lane);
  - ego vs OXTS: yaw MAE lower than iter13 on all 6 KITTI clips (CPU: 0.12-1.12 deg/s vs 0.51-24.4); speeds expected
    (CPU): sample 25.0 (prior), exit ~25.6, late ~18.5, final ~20.3, cyclist ~10.8, lane_split ~4.6 m/s;
    lane pitch applied on sample / exit / late / final / cyclist only (theta -0.38 .. +0.13 deg).
  - If any GPU ego number deviates by > 5 % from the CPU expectation, rerun that clip with `FBEV_EGO_PITCH=0` and report both.
- **Deliver:** SHA256SUMS.txt, run_meta.json, metrics.md/json, metrics_vs_iter13.md, ego_eval_oxts.json/txt, gate.jsonl,
  car_audit, ego_speed_table.md, and stills / videos to the private Drive folder as for iter13 (public clips only).
