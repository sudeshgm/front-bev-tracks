# iter10 replay vs iter9 (and YOLOPv2)

Replay on the iter9 / iter8 GPU outputs (CPU, no SAM re-run). `d` = iter10 replay - iter9 replay. YOLOPv2 is a measurement baseline only (not in the pipeline).

## lanes_10_wjit

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
|---|---|---|---|---|---|
| a2d2a |  |  |  |  |  |
| a2d2b |  |  |  |  |  |
| kitti_city_cyclist_candidate | 0.005 | 0.007 | 0.007 | 0.004 | -0.003 |
| kitti_highway_exit_approach | 0.007 | 0.023 | 0.023 | 0.017 | -0.006 |
| kitti_highway_final | 0.008 | 0.013 | 0.013 | 0.012 | -0.001 |
| kitti_highway_late | 0.009 | 0.02 | 0.02 | 0.019 | -0.001 |
| kitti_highway_sample | 0.006 | 0.006 | 0.006 | 0.004 | -0.002 |
| kitti_lane_split_candidate | 0.092 | 0.131 | 0.136 | 0.133 | -0.003 |
| munich |  | 0.006 | 0.006 | 0.006 | +0.000 |

## lanes_L10_jit2

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.008 | 0.008 | 0.008 | +0.000 |
| a2d2b |  | 0.012 | 0.012 | 0.012 | +0.000 |
| kitti_city_cyclist_candidate | 0.001 | 0.003 | 0.002 | 0.002 | +0.000 |
| kitti_highway_exit_approach | 0.013 | 0.005 | 0.005 | 0.005 | +0.000 |
| kitti_highway_final | 0.024 | 0.012 | 0.013 | 0.015 | +0.002 |
| kitti_highway_late | 0.014 | 0.012 | 0.012 | 0.011 | -0.001 |
| kitti_highway_sample | 0.019 | 0.005 | 0.005 | 0.003 | -0.002 |
| kitti_lane_split_candidate | 0.257 | 0.011 | 0.011 | 0.012 | +0.001 |
| munich |  | 0.003 | 0.003 | 0.003 | +0.000 |

## lanes_R10_jit2

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.007 | 0.007 | 0.007 | +0.000 |
| a2d2b |  | 0.035 | 0.035 | 0.035 | +0.000 |
| kitti_city_cyclist_candidate | 0.004 | 0.006 | 0.006 | 0.003 | -0.003 |
| kitti_highway_exit_approach | 0.009 | 0.018 | 0.018 | 0.014 | -0.004 |
| kitti_highway_final | 0.017 | 0.004 | 0.004 | 0.004 | +0.000 |
| kitti_highway_late | 0.014 | 0.007 | 0.008 | 0.007 | -0.001 |
| kitti_highway_sample | 0.005 | 0.003 | 0.004 | 0.004 | +0.000 |
| kitti_lane_split_candidate | 0.721 | 0.022 | 0.016 | 0.012 | -0.004 |
| munich |  | 0.002 | 0.002 | 0.002 | +0.000 |

## lanes_L10_pres

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 15.1 | 15.1 | 15.1 | +0.000 |
| a2d2b |  | 8.1 | 8.1 | 8.1 | +0.000 |
| kitti_city_cyclist_candidate | 100.0 | 93.3 | 93.3 | 100.0 | +6.700 |
| kitti_highway_exit_approach | 100.0 | 95.0 | 95.0 | 100.0 | +5.000 |
| kitti_highway_final | 100.0 | 93.3 | 93.3 | 93.3 | +0.000 |
| kitti_highway_late | 100.0 | 92.5 | 92.5 | 97.5 | +5.000 |
| kitti_highway_sample | 100.0 | 95.0 | 95.0 | 100.0 | +5.000 |
| kitti_lane_split_candidate | 76.7 | 93.3 | 93.3 | 100.0 | +6.700 |
| munich |  | 85.7 | 85.7 | 85.9 | +0.200 |

## lanes_R10_pres

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 25.0 | 25.0 | 26.7 | +1.700 |
| a2d2b |  | 2.3 | 2.3 | 2.3 | +0.000 |
| kitti_city_cyclist_candidate | 100.0 | 93.3 | 93.3 | 100.0 | +6.700 |
| kitti_highway_exit_approach | 100.0 | 95.0 | 95.0 | 100.0 | +5.000 |
| kitti_highway_final | 100.0 | 90.0 | 90.0 | 90.0 | +0.000 |
| kitti_highway_late | 100.0 | 95.0 | 95.0 | 100.0 | +5.000 |
| kitti_highway_sample | 100.0 | 95.0 | 95.0 | 100.0 | +5.000 |
| kitti_lane_split_candidate | 26.7 | 73.3 | 73.3 | 73.3 | +0.000 |
| munich |  | 65.2 | 65.2 | 65.4 | +0.200 |

## lanes_12_wjit

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.015 | 0.015 | 0.015 | +0.000 |
| a2d2b |  | 0.046 | 0.046 | 0.046 | +0.000 |
| kitti_city_cyclist_candidate | 0.004 | 0.006 | 0.006 | 0.002 | -0.004 |
| kitti_highway_exit_approach | 0.01 | 0.021 | 0.022 | 0.016 | -0.006 |
| kitti_highway_final | 0.012 | 0.017 | 0.017 | 0.02 | +0.003 |
| kitti_highway_late | 0.01 | 0.02 | 0.02 | 0.018 | -0.002 |
| kitti_highway_sample | 0.006 | 0.005 | 0.006 | 0.004 | -0.002 |
| kitti_lane_split_candidate | 0.219 | 0.082 | 0.073 | 0.08 | +0.007 |
| munich |  | 0.006 | 0.006 | 0.006 | +0.000 |

## edges_10_wjit

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.07 | 0.075 | 0.056 | -0.019 |
| a2d2b |  | 0.057 | 0.085 | 0.079 | -0.006 |
| kitti_city_cyclist_candidate |  | 0.054 | 0.054 | 0.054 | +0.000 |
| kitti_highway_exit_approach |  | 0.048 | 0.048 | 0.048 | +0.000 |
| kitti_highway_final |  | 0.023 | 0.023 | 0.023 | +0.000 |
| kitti_highway_late |  | 0.035 | 0.036 | 0.036 | +0.000 |
| kitti_highway_sample |  | 0.019 | 0.019 | 0.019 | +0.000 |
| kitti_lane_split_candidate |  | 0.125 | 0.129 | 0.129 | +0.000 |
| munich |  | 0.043 | 0.042 | 0.038 | -0.004 |

## edges_L10_jit2

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.038 | 0.039 | 0.041 | +0.002 |
| a2d2b |  | 0.038 | 0.052 | 0.05 | -0.002 |
| kitti_city_cyclist_candidate |  | 0.018 | 0.022 | 0.022 | +0.000 |
| kitti_highway_exit_approach |  | 0.016 | 0.014 | 0.014 | +0.000 |
| kitti_highway_final |  | 0.029 | 0.03 | 0.03 | +0.000 |
| kitti_highway_late |  | 0.02 | 0.017 | 0.017 | +0.000 |
| kitti_highway_sample |  | 0.02 | 0.019 | 0.019 | +0.000 |
| kitti_lane_split_candidate |  | 0.042 | 0.041 | 0.041 | +0.000 |
| munich |  | 0.01 | 0.01 | 0.01 | +0.000 |

## edges_R10_jit2

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 0.032 | 0.039 | 0.034 | -0.005 |
| a2d2b |  | 0.032 | 0.037 | 0.034 | -0.003 |
| kitti_city_cyclist_candidate |  | 0.01 | 0.01 | 0.01 | +0.000 |
| kitti_highway_exit_approach |  | 0.024 | 0.024 | 0.024 | +0.000 |
| kitti_highway_final |  | 0.025 | 0.025 | 0.025 | +0.000 |
| kitti_highway_late |  | 0.015 | 0.019 | 0.019 | +0.000 |
| kitti_highway_sample |  | 0.009 | 0.01 | 0.01 | +0.000 |
| kitti_lane_split_candidate |  | 0.042 | 0.037 | 0.034 | -0.003 |
| munich |  | 0.024 | 0.022 | 0.021 | -0.001 |

## edges_L10_pres

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 95.3 | 97.1 | 97.1 | +0.000 |
| a2d2b |  | 97.1 | 97.1 | 97.1 | +0.000 |
| kitti_city_cyclist_candidate | 0.0 | 93.3 | 93.3 | 93.3 | +0.000 |
| kitti_highway_exit_approach | 0.0 | 95.0 | 95.0 | 95.0 | +0.000 |
| kitti_highway_final | 0.0 | 90.0 | 90.0 | 90.0 | +0.000 |
| kitti_highway_late | 0.0 | 95.0 | 95.0 | 95.0 | +0.000 |
| kitti_highway_sample | 0.0 | 95.0 | 95.0 | 95.0 | +0.000 |
| kitti_lane_split_candidate | 0.0 | 93.3 | 93.3 | 93.3 | +0.000 |
| munich |  | 99.8 | 99.8 | 99.8 | +0.000 |

## edges_R10_pres

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
|---|---|---|---|---|---|
| a2d2a |  | 94.8 | 94.8 | 94.8 | +0.000 |
| a2d2b |  | 97.1 | 97.1 | 97.1 | +0.000 |
| kitti_city_cyclist_candidate | 0.0 | 93.3 | 93.3 | 93.3 | +0.000 |
| kitti_highway_exit_approach | 0.0 | 95.0 | 95.0 | 95.0 | +0.000 |
| kitti_highway_final | 0.0 | 86.7 | 86.7 | 86.7 | +0.000 |
| kitti_highway_late | 0.0 | 87.5 | 87.5 | 87.5 | +0.000 |
| kitti_highway_sample | 0.0 | 95.0 | 95.0 | 95.0 | +0.000 |
| kitti_lane_split_candidate | 0.0 | 93.3 | 93.3 | 93.3 | +0.000 |
| munich |  | 84.1 | 84.3 | 84.3 | +0.000 |

## accel_rms_mps2

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
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

## id_switch_proxy_per100f

| clip | YOLOPv2 | iter9 GPU | iter9 replay | iter10 replay | d |
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
