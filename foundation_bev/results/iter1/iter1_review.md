# Foundation BEV iter1 review

Reviewed 23 Nov 2026 (IST). Inputs: `stills/` (27 viewed in part: kitti_highway_sample_f0020, kitti_highway_exit_approach_f0020,
kitti_city_cyclist_candidate_f0015, kitti_lane_split_candidate_f0006, munich_f0450, munich_f0858, a2d2a_f0086),
`polylines/*_polylines.json`, `run_meta.json`, GPU Bot's `src/foundation_bev.py` / `kernel/run.py` / `ret/log_stage1.txt`.
Stats script: `/workspace/foundation_bev/iter2_spec/analyze_polylines.py` (re-run: `cd iter1 && python3 ../iter2_spec/analyze_polylines.py`).

## Verdict

Not comparable with the YOLO overlays yet. There are **no objects or trajectories at all**: WildDet3D stage 1 died on
`ModuleNotFoundError: sam3` and `| tee` hid the rc. The lines come from the fallback route only and carry four
systematic bugs, which matter more than the missing smoothing:

1. **Hook artefact in 100 % of polylines** (lanes and edges, every clip). `smooth_poly()` uses `np.convolve(..., mode="same")`
   with zero padding, then restores only the end points. So the 2nd and 2nd-last points are pulled about 20 % towards the image
   origin (0,0). That produces the long diagonal strokes toward the top-left seen in every still (Munich "lanes" in the sky,
   edge hooks at the far end) and spurious BEV segments.
2. **Munich horizon is wrong.** The road vanishing row is about 552 px, but the run assumed cy = 360 and road band 0.48–0.96
   (starting at 345 px). The first about 200 px of the "road band" are sky and buildings, the SAM road mask wanders into trees
   and wires, and every BEV range is badly mis-scaled. Implied edge-to-edge width at z = 10 m is 3.3 m for a multi-lane road.
3. **The lane mask sits outside the SAM road.** Bright-stripe lanes are intersected with an eroded road mask. SAM's asphalt
   mask has holes or ends at the paint, so the real markings are mostly removed, and dashed lines fail the 36 px
   vertical-span filter. That leaves 45–93 % of frames with no lane, and the green that does survive is often grass or shadow
   (exit_approach).
4. **Edges = first/last road-mask column per row.** This jumps across parked cars, the tram track, the bonnet and the image
   border (vertical stubs at x = 0). There is no temporal model, and the per-frame DAV2 scale changes the lift every frame.

Also: stride frames reuse the previous frame's detections (`dets_for_frame`), which would step the tracks once WD3D runs.
Per-frame DAV2 scale has IQR of roughly 0.8–3.8 (CPU SAM2.1-S + DAV2-S test, iter2_spec/metrics_cpu_models.md). The A2D2 burned-in overlays (yellow boxes, coloured LiDAR dots, red
centre line) are inside the road band. The HUD text reads `projection=physical_ai_calibration:camera_front_wide_120fov`, so
the "A2D2" crops look like PhysicalAI-AV visualisations (NYC yellow cab in a2d2b), not raw A2D2.

## Per-clip issues (short)

- **kitti_highway_sample**: cleanest clip. Edges follow the carriageway but have end hooks (both sides). Lanes are missing in
  45 % of frames, and the dashed centre line is never found (span filter).
- **kitti_highway_exit_approach**: about 5 green strokes per frame, mostly shadow and grass, not paint. Left-lane jitter is
  0.87 m median / 2.3 m p90 at z = 10 m.
- **kitti_highway_late / final**: edges are OK apart from the hooks. Lanes are present in only 10–27 % of frames, with jumps
  up to 2.4 m at z = 20 m (final, right).
- **kitti_city_cyclist_candidate**: the left edge zig-zags along parked cars and shadows (p90 2.7 m at z = 10 m). The right
  edge follows the tram-track kerb (fine), but there is a hook to the image border. The cyclist and car are untracked.
- **kitti_lane_split_candidate**: the worst edges. The road mask boundary runs through the cyclist and the parked lorry
  (zig-zag in BEV, jitter 0.4 m median / 3.8 m p90). Lanes are absent in 93 % of frames.
- **munich**: horizon or band wrong (see 2). Edges and "lanes" are drawn in the sky and over wires, parked cars and tram
  rails. Lanes are absent in 83 % of frames.
- **a2d2a / a2d2b**: 30 % (a) / 14 % (b) of edge pixels and 43 % / 30 % of lane pixels are above the horizon (the band starts
  at 0.32·h < cy). Lanes are absent in 75 % / 87 % of frames. The HUD overlay contaminates the masks.

## Key numbers

Jitter = median (and p90) |frame-to-frame change| of the nearest left/right line's lateral offset at z (BEV metres).
hook % = polylines whose 2nd/2nd-last point is > 40 px off the chain. zig-zag % = BEV steps with |dx| > 1 m and |dx| > |dz|
(proxy for an edge cutting across an obstacle). Note fps: KITTI 4, A2D2 10, Munich 30.

| clip | fps | frames | lane missing % | lanes/frame | edge missing % | edge hook % | edge zig-zag % | pts above horizon % (lane/edge) | edge width z10 (m) | edge jit L/R z10 (m, med/p90) | edge jit L/R z20 (m, med/p90) | lane jit L/R z10 (m) | lane pres L/R z10 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2d2a | 10 | 172 | 75.0 | 0.37 | 0.0 | 100.0 | 0.8 | 42.6/30.1 | 8.67 | 0.07/0.48 · 0.07/0.21 | 0.23/2.96 · 0.11/1.92 | –/– | 2.9/1.7 |
| a2d2b | 10 | 172 | 87.2 | 0.19 | 0.0 | 100.0 | 1.4 | 30.2/14.3 | 10.75 | 0.25/0.94 · 0.10/0.71 | 0.29/4.02 · 0.15/1.69 | –/– | 0.0/0.0 |
| kitti_city_cyclist_candidate | 4 | 30 | 73.3 | 0.27 | 0.0 | 100.0 | 0.0 | 0.0/0.9 | 4.50 | 0.24/2.67 · 0.02/0.09 | 0.16/0.91 · 0.02/0.05 | 0.40/– | 26.7/0.0 |
| kitti_highway_exit_approach | 4 | 40 | 2.5 | 5.35 | 0.0 | 100.0 | 0.0 | 0.7/0.9 | 5.63 | 0.38/1.77 · 0.06/0.49 | 0.09/0.26 · 0.17/2.72 | 0.87/0.31 | 95.0/60.0 |
| kitti_highway_final | 4 | 30 | 60.0 | 0.83 | 0.0 | 100.0 | 0.1 | 3.4/0.9 | 6.37 | 0.07/0.29 · 0.07/0.52 | 0.10/0.44 · 0.06/0.23 | 0.73/0.25 | 13.3/26.7 |
| kitti_highway_late | 4 | 40 | 67.5 | 0.4 | 0.0 | 100.0 | 0.1 | 6.8/0.9 | 5.60 | 0.13/0.61 · 0.06/0.18 | 0.11/0.23 · 0.10/0.31 | 0.14/0.43 | 12.5/10.0 |
| kitti_highway_sample | 4 | 40 | 45.0 | 0.55 | 0.0 | 100.0 | 0.0 | 0.0/0.8 | 8.88 | 0.08/0.33 · 0.04/2.44 | 0.06/0.15 · 0.05/0.21 | –/0.15 | 0.0/55.0 |
| kitti_lane_split_candidate | 4 | 30 | 93.3 | 0.07 | 0.0 | 100.0 | 0.5 | 0.0/1.0 | 4.18 | 0.42/0.93 · 0.21/1.16 | 0.44/1.38 · 0.38/3.75 | –/– | 6.7/0.0 |
| munich | 30 | 900 | 82.7 | 0.26 | 0.0 | 100.0 | 1.8 | 8.2/3.6 | 3.27 | 0.02/0.93 · 0.01/0.50 | 0.18/2.82 · 0.03/0.81 | 0.00/0.00 | 8.3/0.4 |

Tracks: all 9 CSVs are header-only (0 rows), so there is no acceleration or ID-switch number for iter1.

### What the iter2 line smoother does to the same raw iter1 data

`iter2_spec/replay_iter1_lines.py` replays iter1's own per-frame BEV polylines through the iter2 RANSAC poly2 + EMA 0.3 +
0.5 m gate + 15-frame hold + centred 7-frame window. It uses the same raw data and none of the iter2 extraction or ground
fixes. Edge jitter at z = 10 m drops 6–50× on most clips (for example exit_approach L 0.38 → 0.008 m,
city_cyclist L 0.24 → 0.015 m). kitti_highway_late improves only 0.13 → 0.09 m. The price is presence on clips whose raw edges are
mostly junk (lane_split 10/73 %, a2d2 10–37 %, munich 56/74 %): the gate correctly refuses them. Replayed lanes are almost
empty, because iter1's raw lane strokes are too short or sparse to fit. Lane presence needs the iter2 extraction changes
(filled road mask, top-hat stripes, BEV clustering across dashes), not just smoothing.

| clip | iter1 edge jit L/R z10 (m) | replay edge jit L/R z10 (m) | iter1 edge jit L/R z20 | replay edge jit L/R z20 | replay edge presence L/R z10 % |
|---|---|---|---|---|---|
| a2d2a | 0.07/0.07 | 0.00/0.01 | 0.23/0.11 | 0.01/0.02 | 10.5/37.2 |
| a2d2b | 0.25/0.10 | 0.00/0.01 | 0.29/0.15 | 0.01/0.01 | 20.3/30.2 |
| kitti_city_cyclist_candidate | 0.24/0.02 | 0.01/0.01 | 0.16/0.02 | 0.02/0.02 | 93.3/93.3 |
| kitti_highway_exit_approach | 0.38/0.06 | 0.01/0.01 | 0.09/0.17 | 0.01/0.01 | 95.0/87.5 |
| kitti_highway_final | 0.07/0.07 | 0.01/0.01 | 0.10/0.06 | 0.01/0.01 | 86.7/83.3 |
| kitti_highway_late | 0.13/0.06 | 0.09/0.07 | 0.11/0.10 | 0.04/0.09 | 95.0/95.0 |
| kitti_highway_sample | 0.08/0.04 | 0.01/0.01 | 0.06/0.05 | 0.01/0.00 | 95.0/92.5 |
| kitti_lane_split_candidate | 0.42/0.21 | 0.07/0.00 | 0.44/0.38 | 0.01/0.00 | 10.0/73.3 |
| munich | 0.02/0.01 | 0.00/0.01 | 0.18/0.03 | 0.00/0.02 | 55.9/73.7 |

## YOLO baseline

No YOLO track or polyline dumps exist on the box. Only rendered videos exist: `gpubot/drive-videos-22nov/final/yolopv2_{munich,a2d2a,a2d2b}.mp4`
plus the sam-bev / front-bev-tracks overlay script. Its smoothing (`front-bev-tracks lane-solver-fast`, `scripts/overlay.py` +
`qdtrack/tracks.py`) is offline:
- objects: constant-velocity Kalman (sigma_a = 3 m/s²) run forward and then an RTS backward pass. The along-range
  measurement sigma grows with range² (`min(6, max(1, d²·5/590))`), parked cars are held at their median position, and
  passing cars are held at their head-on position. Tracks need ≥ 3 hits and the trail is the last 40 samples.
- lanes: whole-clip world-frame curve fitting (`fit_lane_curves`): every frame's pixels are fitted to one static world
  curve per marking through the ego poses, then drawn in each frame's ego frame. Tracks need ≥ 4 hits.
  That is why YOLO lines look rock-steady: they are one static fit per marking, not per-frame detections.
So "YOLO-level smoothness" means offline smoothing. iter2 adopts RTS for tracks and a centred (zero-phase) window over the
EMA'd line coefficients. To score YOLO with the same metric, `iter2_spec/yolo_baseline/overlay_dump.patch` adds `--dump-dir`
to `overlay.py` (it applies cleanly to the sam-bev-22nov repo copy).
