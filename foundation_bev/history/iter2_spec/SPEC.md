# Foundation BEV iter2 spec (WildDet3D + SAM 2.1 + Depth Anything V2, no YOLO or TwinLiteNet)

Review of iter1: ../iter1/iter1_review.md. Full diff against GPU Bot's iter1 code: patch.diff.
Status: CPU-tested only. Nothing has been launched. GPU Bot runs this only after Sudesh says yes.

## Dataset contents (Kaggle dataset, free T4)
- videos/*.mp4 (same 9 clips as iter1) and motion_assumptions.json
- cams_iter2.json: replaces cams.json. Munich horizon_v = 552 with road band 0.78–0.95. Rails are on for munich and kitti_city_cyclist only.
- src/: foundation_bev.py, bev_smooth.py, wd3d_infer.py, smoothness_metrics.py, analyze_polylines.py
- Kernel: src/run_iter2.py (replaces kernel/run.py). It writes `{tag}_foundation_bev_iter2.mp4`, so iter1 outputs are never overwritten.

## (a) WildDet3D stage-1 fix (iter1 failed with ModuleNotFoundError: sam3, and the pipeline kept going on 0 detections)
- `git submodule update --init --recursive`, then pin checks:
  - sam3_for_wilddet3d = 159490bb13108749dd5ceff68f42cd1792e60641
  - lingbot_depth = ff29de5…
  - Mismatch exits rc 21.
- Every shell step runs under `bash -o pipefail`, so tee no longer masks failures.
- Pre-flight import with PYTHONPATH = third_party/sam3 + lingbot + root (rc 26). wd3d_infer itself exits 11 if `import sam3` fails.
- 3-frame WD3D pre-flight on kitti_highway_sample + munich (rc 27).
- Stage 1 must print `STAGE1 OK` (rc 23). wd3d_infer exits 12 if any run has 0 object boxes.
- Stage 2 `--require-dets` exits 7 if a clip has no detection file.
- sam3 is code-only. It is WD3D's internal dependency; SAM 3 is not run as a segmenter (SAM 2.1 does the road).

## (b) Tracking (bev_smooth.ObjTracker)
- ByteTrack two-stage association: high ≥ 0.5, then low 0.25–0.5. Cost is IoU + BEV distance, solved with Hungarian. car and person are never merged.
- Constant-velocity Kalman in the world frame (s = z + ego travel).
  - sigma_a: car 2.5, person 0.8, bicycle 1.5.
  - Measurement noise: lateral 0.15 + 0.02z; longitudinal clip(0.05z + 0.004z², 0.3, 6).
- Coast up to 10 frames; keep a track only if it has ≥ 5 hits; class by majority vote.
- RTS smoother over the whole track. Stationary hold: speed < 0.8 m/s → median position.
- Re-link fragments: gap ≤ 10 frames and either distance ≤ 1.5 m, or Mahalanobis² ≤ 9.21 with distance ≤ max(4, 0.15z).

## (c) Lanes and edges (bev_smooth.LineTracker, ransac_poly2)
- Per frame:
  - RANSAC fit of BEV x(z). Degree 1 if the z span is ≤ 12 m. Limits |c1| ≤ 0.4, |c2| ≤ 0.012.
  - Side test at the near end of the fit, not at z = 0.
- Temporal:
  - EMA on coefficients, alpha 0.3.
  - Gate 0.5 m. Jumps ≤ 1.5 m re-init only after 3 consistent frames.
  - Hold the last fit 15 frames. Display only after 3 hits.
  - Centred 7-frame zero-phase window (offline, same idea as YOLO's offline RTS).
  - Motion model "follow" by default ("straight" only with real odometry).
- Edges:
  - Lower envelope of the SAM 2.1 drivable mask (closed, holes filled).
  - Masked by WD3D object boxes and by the burned-in HUD overlay mask.
  - Rail rejection (Hough pairs at 1.30–1.60 m gauge), only on clips with rails on.
- Lanes:
  - WD3D 'lane marking' boxes + top-hat bright mask inside the horizontally dilated road.
  - Skeletonised, then clustered by lateral offset.

## (d) Ground plane and fx
- Per-clip ground: median over frames of the VP / DAV2-plane pitch. In auto mode it is 0 for KITTI-calibrated clips, else the estimate clamped to ±3°. A cam `horizon_v` value overrides it (Munich 552 px, about −11.9°).
- A2D2 fx:
  - On flat ground, lateral X = (u−cx)·H/(v−v_h) does not depend on fx. So lane width ≈ 3.5 m checks camera height and horizon, not fx; fx only scales range.
  - Keep 66°. Optional sweep: `--hfov-deg 66/90/120`, compared via `static_closing_ratio` in meta. The HUD text says 120fov, but these look like PhysicalAI-AV renders.

## (e) Metrics (smoothness_metrics.py)
- Track accel RMS (smoothed and raw), ID-switch proxy per 100 frames, lane/edge L/R jitter and presence at z = 10/20 m, and coefficient jitter.
- No YOLO dumps exist. yolo_baseline/overlay_dump.patch adds `--dump-dir` to overlay.py (git apply --check OK). With it, the same script scores YOLO.

## (f) Render
Front + BEV side by side with only:
- class-coloured trajectories
- lane curves
- edge curves (separate colour)
- ego path

No depth maps or edge maps.

## Runtime and failure codes
- Stage 1 is about 2.15 s/frame on T4, so roughly 1 h for 1454 frames. Wall cap 3.8 h.
- rc 7: no detections for a clip
- rc 11: sam3 import failed
- rc 12: 0 object boxes
- rc 21: submodule pin mismatch
- rc 23: no STAGE1 OK
- rc 26: pre-flight import failed
- rc 27: WD3D pre-flight failed

## Test status (CPU, box)
- tests/test_bev_smooth.py: ALL TESTS PASSED.
- smoke/run_smoke.sh (model-free, 9 clips): all rc 0.
- smoke_kernel (`FBEV_SMOKE=1 FBEV_MAXF=12`): rc 0, fails none, 9 iter2 mp4s.
- cpu_models/ (real SAM2.1-S + DAV2-S, no WD3D): all rc 0. Results in cpu_models/metrics_cpu_models.md.

## Known limitations
- No WD3D boxes on CPU, so objects and object masking are untested with real detections.
- Munich nearest visible ground is about 10.4 m, so z = 10 metrics are absent there.
- kitti_lane_split has no lanes, and its right edge is rejected next to the parked lorry.
- a2d2 has no lanes.
- The ego path is drawn straight ahead (no odometry).
