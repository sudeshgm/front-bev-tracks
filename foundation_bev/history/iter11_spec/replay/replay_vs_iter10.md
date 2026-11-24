# iter11 replay vs iter10 (and YOLOPv2)

CPU replay on the iter10 GPU outputs (no SAM re-run): edges = iter10 GPU `edges_raw` through the pipeline's own `StageLines.post_edges` (trim, kerb, hold, veto, gores) + centre smoothing + edge backfill; lanes = iter8 GPU unsmoothed lanes + centre smoothing + pair width + lane backfill. `iter10 replay` = the same harness with the iter11 switches off; it reproduces the iter10 GPU numbers (lanes exactly; edges within 0.002 except lane_split R jit2). `d` = iter11 replay - iter10 replay. YOLOPv2 is a measurement baseline only (not in the pipeline).

## lanes_10_wjit

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  |  |  |  |  |
| a2d2b |  |  |  |  |  |
| kitti_city_cyclist_candidate | 0.005 | 0.004 | 0.004 | 0.001 | -0.003 |
| kitti_highway_exit_approach | 0.007 | 0.017 | 0.017 | 0.007 | -0.010 |
| kitti_highway_final | 0.008 | 0.012 | 0.012 | 0.006 | -0.006 |
| kitti_highway_late | 0.009 | 0.019 | 0.019 | 0.006 | -0.013 |
| kitti_highway_sample | 0.006 | 0.004 | 0.004 | 0.002 | -0.002 |
| kitti_lane_split_candidate | 0.092 | 0.131 | 0.133 | 0.133 | +0.000 |
| munich |  | 0.006 | 0.006 | 0.006 | +0.000 |

## lanes_12_wjit

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.015 | 0.015 | 0.015 | +0.000 |
| a2d2b |  | 0.046 | 0.046 | 0.046 | +0.000 |
| kitti_city_cyclist_candidate | 0.004 | 0.002 | 0.002 | 0.002 | +0.000 |
| kitti_highway_exit_approach | 0.01 | 0.016 | 0.016 | 0.007 | -0.009 |
| kitti_highway_final | 0.012 | 0.02 | 0.02 | 0.011 | -0.009 |
| kitti_highway_late | 0.01 | 0.018 | 0.018 | 0.007 | -0.011 |
| kitti_highway_sample | 0.006 | 0.003 | 0.004 | 0.002 | -0.002 |
| kitti_lane_split_candidate | 0.219 | 0.082 | 0.08 | 0.08 | +0.000 |
| munich |  | 0.006 | 0.006 | 0.006 | +0.000 |

## lanes_L10_jit2

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.008 | 0.008 | 0.008 | +0.000 |
| a2d2b |  | 0.012 | 0.012 | 0.012 | +0.000 |
| kitti_city_cyclist_candidate | 0.001 | 0.002 | 0.002 | 0.002 | +0.000 |
| kitti_highway_exit_approach | 0.013 | 0.005 | 0.005 | 0.004 | -0.001 |
| kitti_highway_final | 0.024 | 0.016 | 0.015 | 0.011 | -0.004 |
| kitti_highway_late | 0.014 | 0.011 | 0.011 | 0.011 | +0.000 |
| kitti_highway_sample | 0.019 | 0.004 | 0.003 | 0.003 | +0.000 |
| kitti_lane_split_candidate | 0.257 | 0.011 | 0.012 | 0.012 | +0.000 |
| munich |  | 0.003 | 0.003 | 0.003 | +0.000 |

## lanes_R10_jit2

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.007 | 0.007 | 0.007 | +0.000 |
| a2d2b |  | 0.035 | 0.035 | 0.035 | +0.000 |
| kitti_city_cyclist_candidate | 0.004 | 0.003 | 0.003 | 0.001 | -0.002 |
| kitti_highway_exit_approach | 0.009 | 0.014 | 0.014 | 0.008 | -0.006 |
| kitti_highway_final | 0.017 | 0.004 | 0.004 | 0.004 | +0.000 |
| kitti_highway_late | 0.014 | 0.007 | 0.007 | 0.011 | +0.004 |
| kitti_highway_sample | 0.005 | 0.003 | 0.004 | 0.004 | +0.000 |
| kitti_lane_split_candidate | 0.721 | 0.022 | 0.012 | 0.012 | +0.000 |
| munich |  | 0.002 | 0.002 | 0.002 | +0.000 |

## lanes_L10_pres

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 15.1 | 15.1 | 15.1 | +0.000 |
| a2d2b |  | 8.1 | 8.1 | 8.1 | +0.000 |
| kitti_city_cyclist_candidate | 100.0 | 100.0 | 100.0 | 100.0 | +0.000 |
| kitti_highway_exit_approach | 100.0 | 100.0 | 100.0 | 100.0 | +0.000 |
| kitti_highway_final | 100.0 | 93.3 | 93.3 | 100.0 | +6.700 |
| kitti_highway_late | 100.0 | 97.5 | 97.5 | 97.5 | +0.000 |
| kitti_highway_sample | 100.0 | 100.0 | 100.0 | 100.0 | +0.000 |
| kitti_lane_split_candidate | 76.7 | 100.0 | 100.0 | 100.0 | +0.000 |
| munich |  | 85.9 | 85.9 | 85.9 | +0.000 |

## lanes_R10_pres

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 26.7 | 26.7 | 26.7 | +0.000 |
| a2d2b |  | 2.3 | 2.3 | 2.3 | +0.000 |
| kitti_city_cyclist_candidate | 100.0 | 100.0 | 100.0 | 100.0 | +0.000 |
| kitti_highway_exit_approach | 100.0 | 100.0 | 100.0 | 100.0 | +0.000 |
| kitti_highway_final | 100.0 | 90.0 | 90.0 | 100.0 | +10.000 |
| kitti_highway_late | 100.0 | 100.0 | 100.0 | 100.0 | +0.000 |
| kitti_highway_sample | 100.0 | 100.0 | 100.0 | 100.0 | +0.000 |
| kitti_lane_split_candidate | 26.7 | 73.3 | 73.3 | 73.3 | +0.000 |
| munich |  | 65.4 | 65.4 | 65.4 | +0.000 |

## edges_10_wjit

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.065 | 0.067 | 0.067 | +0.000 |
| a2d2b |  | 0.057 | 0.057 | 0.057 | +0.000 |
| kitti_city_cyclist_candidate |  | 0.054 | 0.054 | 0.076 | +0.022 |
| kitti_highway_exit_approach |  | 0.048 | 0.048 | 0.047 | -0.001 |
| kitti_highway_final |  | 0.023 | 0.023 | 0.026 | +0.003 |
| kitti_highway_late |  | 0.035 | 0.035 | 0.041 | +0.006 |
| kitti_highway_sample |  | 0.019 | 0.021 | 0.022 | +0.001 |
| kitti_lane_split_candidate |  | 0.127 | 0.129 | 0.127 | -0.002 |
| munich |  | 0.04 | 0.04 | 0.042 | +0.002 |

## edges_L10_jit2

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.036 | 0.036 | 0.035 | -0.001 |
| a2d2b |  | 0.038 | 0.038 | 0.038 | +0.000 |
| kitti_city_cyclist_candidate |  | 0.021 | 0.021 | 0.019 | -0.002 |
| kitti_highway_exit_approach |  | 0.016 | 0.016 | 0.016 | +0.000 |
| kitti_highway_final |  | 0.029 | 0.029 | 0.029 | +0.000 |
| kitti_highway_late |  | 0.02 | 0.02 | 0.019 | -0.001 |
| kitti_highway_sample |  | 0.02 | 0.018 | 0.017 | -0.001 |
| kitti_lane_split_candidate |  | 0.042 | 0.042 | 0.042 | +0.000 |
| munich |  | 0.01 | 0.01 | 0.01 | +0.000 |

## edges_R10_jit2

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.03 | 0.03 | 0.03 | +0.000 |
| a2d2b |  | 0.032 | 0.032 | 0.032 | +0.000 |
| kitti_city_cyclist_candidate |  | 0.01 | 0.01 | 0.01 | +0.000 |
| kitti_highway_exit_approach |  | 0.024 | 0.024 | 0.023 | -0.001 |
| kitti_highway_final |  | 0.025 | 0.025 | 0.022 | -0.003 |
| kitti_highway_late |  | 0.015 | 0.015 | 0.014 | -0.001 |
| kitti_highway_sample |  | 0.009 | 0.01 | 0.01 | +0.000 |
| kitti_lane_split_candidate |  | 0.042 | 0.031 | 0.031 | +0.000 |
| munich |  | 0.022 | 0.021 | 0.021 | +0.000 |

## edges_L10_pres

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 95.9 | 95.9 | 97.7 | +1.800 |
| a2d2b |  | 97.1 | 97.1 | 97.1 | +0.000 |
| kitti_city_cyclist_candidate | 0.0 | 93.3 | 93.3 | 100.0 | +6.700 |
| kitti_highway_exit_approach | 0.0 | 95.0 | 95.0 | 100.0 | +5.000 |
| kitti_highway_final | 0.0 | 90.0 | 90.0 | 100.0 | +10.000 |
| kitti_highway_late | 0.0 | 95.0 | 95.0 | 100.0 | +5.000 |
| kitti_highway_sample | 0.0 | 95.0 | 95.0 | 100.0 | +5.000 |
| kitti_lane_split_candidate | 0.0 | 93.3 | 93.3 | 100.0 | +6.700 |
| munich |  | 99.8 | 99.8 | 100.0 | +0.200 |

## edges_R10_pres

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 94.8 | 94.8 | 94.8 | +0.000 |
| a2d2b |  | 97.1 | 97.1 | 100.0 | +2.900 |
| kitti_city_cyclist_candidate | 0.0 | 93.3 | 93.3 | 100.0 | +6.700 |
| kitti_highway_exit_approach | 0.0 | 95.0 | 95.0 | 100.0 | +5.000 |
| kitti_highway_final | 0.0 | 86.7 | 86.7 | 100.0 | +13.300 |
| kitti_highway_late | 0.0 | 87.5 | 87.5 | 100.0 | +12.500 |
| kitti_highway_sample | 0.0 | 95.0 | 95.0 | 100.0 | +5.000 |
| kitti_lane_split_candidate | 0.0 | 93.3 | 93.3 | 100.0 | +6.700 |
| munich |  | 86.2 | 86.4 | 87.9 | +1.500 |

## accel_rms_mps2

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.495 | 0.495 | 0.495 | +0.000 |
| a2d2b |  | 0.821 | 0.821 | 0.821 | +0.000 |
| kitti_city_cyclist_candidate | 0.491 | 0.548 | 0.548 | 0.548 | +0.000 |
| kitti_highway_exit_approach |  | 0.546 | 0.546 | 0.546 | +0.000 |
| kitti_highway_final | 0.755 | 0.574 | 0.574 | 0.574 | +0.000 |
| kitti_highway_late | 0.495 | 0.534 | 0.534 | 0.534 | +0.000 |
| kitti_highway_sample |  | 0.095 | 0.095 | 0.095 | +0.000 |
| kitti_lane_split_candidate | 0.615 | 0.951 | 0.951 | 0.951 | +0.000 |
| munich |  | 0.655 | 0.655 | 0.655 | +0.000 |

## n_tracks

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 23 | 23 | 23 | +0.000 |
| a2d2b |  | 11 | 11 | 11 | +0.000 |
| kitti_city_cyclist_candidate | 3 | 5 | 5 | 5 | +0.000 |
| kitti_highway_exit_approach | 0 | 2 | 2 | 2 | +0.000 |
| kitti_highway_final | 3 | 3 | 3 | 3 | +0.000 |
| kitti_highway_late | 4 | 5 | 5 | 5 | +0.000 |
| kitti_highway_sample | 0 | 1 | 1 | 1 | +0.000 |
| kitti_lane_split_candidate | 10 | 13 | 13 | 13 | +0.000 |
| munich |  | 87 | 87 | 87 | +0.000 |

## id_switch_proxy_per100f

| clip | YOLOPv2 | iter10 GPU | iter10 replay | iter11 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.0 | 0.0 | 0.0 | +0.000 |
| a2d2b |  | 0.0 | 0.0 | 0.0 | +0.000 |
| kitti_city_cyclist_candidate | 0.0 | 0.0 | 0.0 | 0.0 | +0.000 |
| kitti_highway_exit_approach |  | 0.0 | 0.0 | 0.0 | +0.000 |
| kitti_highway_final | 0.0 | 0.0 | 0.0 | 0.0 | +0.000 |
| kitti_highway_late | 0.0 | 0.0 | 0.0 | 0.0 | +0.000 |
| kitti_highway_sample |  | 0.0 | 0.0 | 0.0 | +0.000 |
| kitti_lane_split_candidate | 0.0 | 0.0 | 0.0 | 0.0 | +0.000 |
| munich |  | 0.0 | 0.0 | 0.0 | +0.000 |

