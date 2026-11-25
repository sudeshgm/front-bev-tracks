# iter14 public real-model CPU check (no GPU run)

Decision: no GPU run for iter14. These are full stage 1 + 2 runs of the real pipeline models (WildDet3D, SAM 2.1 hiera-s at the
same checkpoint / config, Depth-Anything-V2 at the same variant) on CPU, on all 9 public clips, iter14 defaults
(`FBEV_TB_ADAPT=auto`, `FBEV_ROAD_VPLANE=auto` with the night_cast gate, ego14 on). The reference is the same CPU pipeline with
both gates forced off (`FBEV_TB_ADAPT=0 FBEV_ROAD_VPLANE=0`). Gate statistics come from `FBEV_GATE_LOG` (per-call JSONL,
summarised with `tools/gate_summary.py`).

Result: **the night_cast gate never fires on any public clip** (0 of 2908 calls), so the iter14 public output equals the
gates-off output: polylines identical apart from the version string, tracks and ego byte-identical (8 clips compared
directly; munich has no gates-off CPU run and is identical by construction because the gate never fired).

| clip | gate calls | gate on | draft-gate passes (S>70 & hue) | road S max | frame-top V min | polylines / tracks vs gates off | ego source (m/s) |
|---|---:|---:|---:|---:|---:|---|---|
| kitti_highway_sample | 80 | **0** | 0 | 32 | 191 | identical / identical | prior 25.0 (bev_road 0.79 rejected) |
| kitti_highway_exit_approach | 80 | **0** | 80 | 81 | 160 | identical / identical | bev_phase 23.6 |
| kitti_highway_late | 80 | **0** | 12 | 96 | 30 | identical / identical | bev_phase 16.7 |
| kitti_highway_final | 60 | **0** | 40 | 111 | 210 | identical / identical | bev_phase 19.3 |
| kitti_city_cyclist_candidate | 60 | **0** | 0 | 62 | 48 | identical / identical | bev_phase 11.3 |
| kitti_lane_split_candidate | 60 | **0** | 0 | 21 | 42 | identical / identical | bev_phase 4.6 |
| munich | 1800 | **0** | 0 | 32 | 228 | no gates-off reference (identical by construction: gate on 0) | scene 13.1 |
| a2d2a | 344 | **0** | 40 | 107 | 91 | identical / identical | prior 8.0 (bev_road 0.14 rejected) |
| a2d2b | 344 | **0** | 18 | 87 | 99 | identical / identical | prior 8.0 (bev_road 0.34 rejected) |

Columns: *draft-gate passes* = calls that would have passed the earlier hue + S>70 draft rule (rejected because it fired on
daylight clips); *road S max* / *frame-top V min* = extremes of the logged SAM-road HSV stats. The adopted rule is
hue 10-40 AND ((S>80 AND topV<25) OR (S>105 AND topV<90)); the closest public calls are kitti_highway_late (S max 96, topV min 30; never in the same call above both thresholds)
and a2d2a (S <= 107 with topV >= 123).

## Gores (CPU, defaults)
- kitti_highway_exit_approach: f21-26 (same as the iter13 GPU replay).
- munich: f761-765 (iter13 GPU replay: f760-765; one-frame onset difference from the CPU vs GPU SAM backend, not from iter14
  code: the gate never fired, and the iter14 replay on the iter13 GPU dumps reproduces f760-765 byte-identically).
- all other public clips: none.

## Ego speed source (ego14)
bev_road is only consulted when the chain would otherwise fall back to the prior; on sample (0.79 m/s), a2d2a (0.14) and
a2d2b (0.34) it is rejected by the guard (ratio outside 0.33-3x prior), so the prior is kept as in iter13. munich uses
`scene` (450 repeated frames make bev_phase ineligible).

## Private fc1 clips
Aggregate numbers only; see SPEC.md (no frames, images, coordinates or files from those clips are in this repo).
