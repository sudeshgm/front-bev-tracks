# Foundation BEV iter3 spec (draft, not launched)

Base: iter2 (the run in ../iter2). WD3D stage 1 worked on all clips. Filtered accel RMS 0.47–1.70 m/s², ID switches 0.
New modules:
- src/bev_gate.py
- src/ego_motion.py

Tests: tests/test_iter3.py (ALL ITER3 TESTS PASSED).
Offline replay: replay_iter2_tracks.py, with results in replay_iter2_tracks*.md/.json.
patch.diff is the diff against iter2_spec: foundation_bev.py, run_iter2→run_iter3, smoothness_metrics.py, cams_iter3.json, plus the new bev_gate.py, ego_motion.py and run_yolo_baseline.sh.
**Status: wired in. See "Implemented in the iter3 package" at the end.**

## 1. False positives: detection-level gates (bev_gate.gate_detections, per WD3D box before association)
- **Per-class score thresholds:** car 0.35, person 0.45, bicycle 0.50. A track must also have a median hit score at or above its class threshold.
- **Ground contact:**
  - The box bottom must be below the horizon.
  - The flat-ground range at the bottom row must agree with the WD3D 3D centre z (or DAV2 depth) within max(1.5 m, 30%).
  - Metric height and width must fit the class prior: person 1.0–2.2 m tall, car 1.0–3.6, bicycle 0.7–2.2. A box may be up to 25% taller than the maximum; truncated boxes are allowed to be short.
  - This removes signs, posts and billboards that float or are badly sized.
- **Vertical-plane rejection** (person and bicycle):
  - RANSAC a near-vertical plane (|n_y| < 0.35) on DAV2 points in the box plus a ring of 0.8× box size around it.
  - If at least 85% of the box and at least 60% of the ring lie on that plane (residual < 3% of z), the detection is painted on a wall or billboard. This targets Munich f0450.
- **Bicycle:** needs a person box with IoU ≥ 0.05, or a person within 1.2 m in BEV. Otherwise it needs a riderless side-on size (0.8–1.3 m tall, 1.2–2.0 m wide).
- **Track level:**
  - A bicycle track needs rider support in ≥ 30% of its frames.
  - On highway clips, a person or bicycle more than 1 m outside the road edges in most frames is dropped (guardrail posts, signs). This targets KITTI exit_approach f0020.
- **Hook:** in run_clip2 pass 1 after `frame_objects`, call `BG.gate_detections(dets, gm, depth=dav2_metric, K=(fx,fy,cx,cy))`. Log the reject reasons per clip into meta.

## 2. Static objects and stubs (bev_gate.track_filter, scene_ego_profile)
- **Ego speed from the static majority** (`scene_ego_profile`):
  - For each frame, take the median of -dz/dt (1 s window) over off-lane tracks (|x| ≥ 2 m, z < 40 m) within 0.3–1.8× the prior speed.
  - Then smooth it.
  - The same lift produces both sides of the comparison, so the range-scale error largely cancels.
- **Static rule:** a track is static if its net world-frame speed is < 0.5 m/s + 0.08·v_ego (tolerance for range-scale error). Static tracks are drawn as a dot or short stub, with no tail.
- **Draw gate:** at least max(5, 0.5 s·fps) hits; otherwise the track is not drawn.
- **Hook:** after `ObjTracker.finalize`, call `BG.track_filter`, then draw_bev2 uses the decision.

## 3. Visual ego-motion and a curved ego path (ego_motion.EgoMotion)
- **Method:**
  - Shi-Tomasi features in the SAM road mask, minus object boxes.
  - Forward-backward LK, seeded with flow predicted from candidate speeds (0, 0.3, 0.6, 1.0, 1.4× prior).
  - Lift to BEV and fit a 2D rigid transform with RANSAC (yaw + forward), then median-5 + centred smoothing.
  - The output feeds the curved ego path (`ego_path_in_current_frame`: travelled path plus constant-curvature prediction ahead) and the world frame for static tests.
  - It also motion-compensates the lane EMA: use LineTracker motion "straight" with the measured ds and yaw, replacing "follow".
- **Gating rule:** use VO speed only where its valid fraction is ≥ 0.7 and it agrees with the scene-majority speed within 30%. Otherwise use the scene profile, then the prior. Always use VO yaw rate, but zero it if |yaw| < 0.5°/s for a whole clip.
- **CPU check on the KITTI clips is unresolved:**
  - VO measured 1.6–8 m/s against priors of 10–25 m/s, with valid fractions 0.78–0.97.
  - Highway texture is periodic (dashes, guardrail posts), so LK can alias.
  - The track closing rates in iter2 are also far below the priors, so either the KITTI clips' fps/speed priors are wrong or both cues alias.
  - Resolve before trusting either: compare against KITTI oxts if the source sequence can be identified, or check a dash period of 6 m dash + 12 m gap by hand.
  - Munich scene speed (median 12.7 m/s, p10–p90 7.5–14.3) agrees with its prior of 12.

## 4. Urban lanes
- **Munich:** the corrected horizon (row 552) puts the nearest visible ground at about 10.4 m. Lower `road_y1` from 0.95 to 0.985h, so it starts near 8.5 m (the bonnet is not in frame). Report lane metrics at z = 12 / 20 for Munich; z = 10 is unobservable.
- **Lane prompts:** use the WD3D 'lane marking' boxes as SAM 2.1 box prompts, so the mask is the paint, not the road. Intersect with the dilated road.
- **Dashed-marking aggregation:**
  - Lift each frame's lane skeleton points to the world frame using ego motion (VO or scene speed).
  - Keep a 6-frame (KITTI 4 fps) / 15-frame (30 fps) ring buffer and fit on the union, so dashes plus gaps become a continuous line.
  - Points older than the buffer, or more than 0.4 m from the current fit, are dropped.
- **lane_split:** the iter2 left lane is 0%, and the CPU stills show no painted left line (kerb only). So "lane L" is expected to stay empty; only the edge applies. The edges are 0% because the right edge sits along the parked lorry and the left kerb curves. Changes:
  - Use WD3D car boxes as occluders.
  - Allow deg-2 |c2| ≤ 0.03 for edges when z span ≥ 10 m (urban kerbs curve).
- **a2d2a/b:** lanes < 10%. The HUD centre line dominates, and the paint is faint at 994×334. Keep the HUD mask, and rely on lane-marking box prompts plus aggregation.

## 5. kitti_highway_late "jitter"
- Not noise. The ego is changing lanes: the lanes drift steadily (lane 2 at z = 10 goes -1.59 → +0.53 m over 36 frames).
- First difference median: 0.035 m (late) vs 0.003–0.008 m (others).
- Second difference median: 0.005 m (late) vs 0.001.
- Metric change: report lateral jitter as the median |2nd difference|, and the first difference minus the common-mode shift of all lines (= ego lateral motion).
- Tuning: lanes that jump 0.7 m (lane 3 at f16→f20) come from the re-init gate. Raise reinit_after from 3 to 4 at 4 fps, and use ego lateral motion from VO to shift the coefficients.

## 6. YOLO baseline (measurement only)
- **Script:** yolo_baseline/run_yolo_baseline.sh REPO WEIGHTS DS OUT [cpu|cuda]. It:
  - applies overlay_dump.patch (--dump-dir) to the sam-bev-22nov overlay pipeline (YOLOPv2 TorchScript),
  - runs the 6 KITTI clips,
  - scores YOLO and iter2 with the same smoothness_metrics.py.
- **Not run on CPU:** no yolopv2.pt weights exist on the box, and the user's usage cap was close. GPU Bot has the weights from the 22nov run. About 220 KITTI frames would also be fine on CPU.
- **Note:** the pipeline is YOLOPv2, not ultralytics YOLOv8n. Use the existing overlay so the baseline equals what was shown before.

## Offline replay of iter2 tracks (detection-level gates need boxes and depth, not dumped in iter2)
See replay_iter2_tracks_scene.md (scene ego speed) and replay_iter2_tracks.md (prior speed).
scene ego speed (speed, n tracks, accepted, prior): {"a2d2a": {"median": 8.0, "p10": 5.2, "p90": 10.1, "valid_frac": 0.66, "prior": 8.0}, "a2d2b": {"median": 6.0, "p10": 4.7, "p90": 6.8, "valid_frac": 0.71, "prior": 8.0}, "kitti_city_cyclist_candidate": {"median": 10.0, "p10": 7.9, "p90": 10.0, "valid_frac": 0.07, "prior": 10.0}, "kitti_highway_exit_approach": {"median": 22.0, "p10": 22.0, "p90": 22.0, "valid_frac": 0.0, "prior": 22.0}, "kitti_highway_final": {"median": 25.0, "p10": 25.0, "p90": 25.0, "valid_frac": 0.0, "prior": 25.0}, "kitti_highway_late": {"median": 25.0, "p10": 25.0, "p90": 25.0, "valid_frac": 0.0, "prior": 25.0}, "kitti_highway_sample": {"median": 25.0, "p10": 25.0, "p90": 25.0, "valid_frac": 0.0, "prior": 25.0}, "kitti_lane_split_candidate": {"median": 18.0, "p10": 12.7, "p90": 18.0, "valid_frac": 0.13, "prior": 18.0}, "munich": {"median": 12.7, "p10": 7.5, "p90": 14.3, "valid_frac": 0.91, "prior": 12.0}}

| clip | tracks | drawn moving | drawn static | drop short | drop score | drop bicycle no rider | drop off-road VRU | accel RMS iter2 (all) | accel RMS iter3 (moving) |
|---|---|---|---|---|---|---|---|---|---|
| a2d2a | 49 | 27 | 9 | 0 | 10 | 3 | 0 | 1.26 | 1.46 |
| a2d2b | 76 | 46 | 10 | 0 | 14 | 6 | 0 | 1.12 | 1.26 |
| kitti_city_cyclist_candidate | 10 | 9 | 0 | 0 | 0 | 1 | 0 | 0.47 | 0.50 |
| kitti_highway_exit_approach | 6 | 2 | 0 | 0 | 1 | 3 | 0 | 0.63 | 0.69 |
| kitti_highway_final | 9 | 5 | 0 | 0 | 3 | 0 | 1 | 0.84 | 0.95 |
| kitti_highway_late | 13 | 8 | 0 | 0 | 2 | 1 | 2 | 1.70 | 1.96 |
| kitti_highway_sample | 5 | 2 | 0 | 0 | 0 | 3 | 0 | 0.50 | 0.69 |
| kitti_lane_split_candidate | 15 | 14 | 0 | 0 | 0 | 1 | 0 | 1.13 | 1.05 |

- accel RMS for iter3 is over moving tails only. It rises slightly because the smooth static tracks leave the pool. Tails drawn per frame drop sharply: in Munich, 108 drawn objects (60 moving + 48 dots) instead of 171 tails.
- The Munich billboard person (track 105, f396–465) is flagged static, so it becomes a dot. Removing it needs the vertical-plane gate (stage 2 with depth).
- exit_approach: all 4 bicycle tracks are dropped (3 have no rider, 1 has a low score).

## Iter3 stage-2 request to GPU Bot (needs Sudesh's yes)
- Dump per-frame WD3D boxes (box, cls, score, 3D centre) and the DAV2 scale per frame, so all gates can be replayed offline next time.
- Run the YOLO baseline script on the KITTI clips.

## Implemented in the iter3 package (draft, not launched)

**Kernel**
- src/run_iter3.py is the kernel (from run_iter2.py).
- New modules copied in: bev_gate.py, ego_motion.py.
- It prefers cams_iter3.json.
- Outputs:
  - results/<clip>_foundation_bev_iter3.mp4
  - dets_dump/<clip>_dets_ego.json
- After stage 2 it prints the `EGO SPEED COMPARISON` table, then runs the optional YOLOPv2 baseline.

**foundation_bev.py hooks**
- **Pass 1**
  - EgoMotion.step on the SAM road mask, with object boxes removed.
  - Per box: DAV2 depth at the box bottom, plus the billboard (vertical-plane) flag for person/bicycle.
  - The dump per frame holds: boxes, scores, b3, depth_z, vplane, lane/edge boxes and the DAV2 scale.
- **Pass 2a**
  - vplane drop, then BG.gate_detections (score, ground contact, size, bicycle rider). Reject counts go into meta.gate_rejects.
  - After the tracker: scene_ego_profile, then select_ego_speed (oxts > flow if within 30% of the scene closing rate > scene > prior).
  - Then BG.track_filter (min hits, score, rider, off-road VRU on cams with highway=true, static).
  - Dropped tracks are removed. The decision is in the CSV column `draw`, with counts in meta.track_decisions.
- **Lines**
  - LaneAggregator: world-frame dashed aggregation, 6 frames at ≤ 10 fps and 15 at 30 fps (`--lane-agg-frames`, -1 = off).
  - Edge curvature cap is per cam (`max_edge_c2` 0.03 for munich, lane_split, cyclist).
  - Munich road_y1 is 0.985.
- **Render**
  - Static tracks: dot + 3-frame stub.
  - Ego path: 2 s travelled path + 2 s constant-curvature prediction from the VO pose.
- **Flags:** `--no-gate`, `--lane-agg-frames`, `--max-edge-c2`, `--dets-json`.

**Metrics.** smoothness_metrics.py gains:
- jit2: median |2nd difference|, lane-change aware.
- wjit: L-R half-spacing change, common mode removed.
- z = 12 columns for Munich.
- On iter2 data, kitti_highway_late lanes go from jit 0.063 to jit2 0.006 m, in line with the other clips (0.001–0.003).

**Not done / limits**
- The lane EMA is not motion-compensated with VO: line_motion stays "follow". VO speed is unverified on KITTI, and a wrong ds would bias the lines. Turn it on only when the selected source is oxts or flow.
- WD3D 'lane marking' boxes are not yet used as SAM box prompts. They are dumped, so they can be tested offline next.

**oxts**
- No KITTI raw oxts on the box, and the clips' source drives are not recorded (STATE.md notes only "KITTI x6 ... squeezed 640x480 at 4 fps"). No comparison was possible.
- To use oxts, set `oxts_speed_mps` per clip in cams_iter3.json.
- CPU smoke: flow gave 0.3–4.8 m/s on KITTI against priors of 10–25. The scene closing rate is unavailable in smoke (synthetic tracks), so the prior was kept. The GPU run will print the real three-way comparison.

**YOLOPv2 baseline**
- `yolo_baseline/run_yolo_baseline.sh REPO WEIGHTS DS OUT [DEVICE] [CAMS_JSON] [COMPARE_RUN]`:
  - REPO may be the front-bev-tracks dir or sam-bev-22nov `ds/fbt_c2b386b.tar.gz`.
  - It applies overlay_dump.patch (dry-run OK on that tarball) and writes motion CSV + camera JSON per clip.
  - It runs overlay.py, which uses YOLOPv2 on CPU via torch.jit, then scores with the same metrics.
- In the kernel it runs when `FBEV_YOLOPV2` (weights) and `FBEV_YOLO_REPO` are set.
- It was tested on the box up to weights loading; it fails there because the box has no yolopv2.pt.

**Expected GPU time (T4)**
- iter2 took 63 min (stage 1: 45 min, stage 2: 13 min).
- iter3 adds LK flow and the plane test (≈ +2–5 min), plus about 5 min of CPU YOLOPv2 for about 220 KITTI frames.
- Total ≈ 1.2 h, under the 3.8 h wall cap.

**Tests**
- tests/test_iter3.py, tests/test_iter3_hooks.py and tests/test_bev_smooth.py: all pass.
- Kernel smoke `FBEV_SMOKE=1 FBEV_MAXF=12`: rc 0, fails none, 9 iter3 mp4s and 9 dets dumps.
