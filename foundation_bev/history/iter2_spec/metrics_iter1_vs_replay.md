# Smoothness metrics

accel = RMS BEV acceleration of drawn track positions (m/s^2); idsw = probable ID switches /100 frames;
jit = median |frame-to-frame change| of nearest left/right line lateral offset at z (m); pres = % frames present.

## iter1_raw

| clip | fps | frames | n_tracks | accel_rms_mps2 | accel_rms_raw_mps2 | id_switch_proxy_per100f | lanes_L10_jit | lanes_R10_jit | lanes_L20_jit | lanes_R20_jit | lanes_L10_pres | lanes_R10_pres | edges_L10_jit | edges_R10_jit | edges_L20_jit | edges_R20_jit | edges_L10_pres | edges_R10_pres |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2d2a | 10.0 | 172 | 0 |  |  |  |  |  | 0.751 | 0.099 | 2.9 | 1.7 | 0.068 | 0.071 | 0.23 | 0.115 | 23.8 | 51.7 |
| a2d2b | 10.0 | 172 | 0 |  |  |  |  |  | 0.772 | 0.145 | 0.0 | 0.0 | 0.255 | 0.104 | 0.287 | 0.151 | 19.8 | 36.0 |
| kitti_city_cyclist_candidate | 4.0 | 30 | 0 |  |  |  | 0.4 |  |  |  | 26.7 | 0.0 | 0.241 | 0.021 | 0.159 | 0.02 | 100.0 | 96.7 |
| kitti_highway_exit_approach | 4.0 | 40 | 0 |  |  |  | 0.872 | 0.308 | 1.159 | 0.103 | 95.0 | 60.0 | 0.376 | 0.06 | 0.091 | 0.168 | 100.0 | 100.0 |
| kitti_highway_final | 4.0 | 30 | 0 |  |  |  | 0.734 | 0.252 | 0.051 | 2.392 | 13.3 | 26.7 | 0.074 | 0.072 | 0.103 | 0.065 | 100.0 | 100.0 |
| kitti_highway_late | 4.0 | 40 | 0 |  |  |  | 0.136 | 0.427 | 0.746 |  | 12.5 | 10.0 | 0.13 | 0.06 | 0.115 | 0.103 | 100.0 | 100.0 |
| kitti_highway_sample | 4.0 | 40 | 0 |  |  |  |  | 0.148 | 0.03 |  | 0.0 | 55.0 | 0.085 | 0.04 | 0.055 | 0.049 | 100.0 | 100.0 |
| kitti_lane_split_candidate | 4.0 | 30 | 0 |  |  |  |  |  |  |  | 6.7 | 0.0 | 0.416 | 0.212 | 0.441 | 0.379 | 100.0 | 100.0 |
| munich | 30.0 | 900 | 0 |  |  |  | 0.001 | 0.001 | 0.005 | 0.001 | 8.3 | 0.4 | 0.019 | 0.01 | 0.181 | 0.028 | 94.0 | 92.0 |

## iter1_raw_through_iter2_line_smoother

| clip | fps | frames | n_tracks | accel_rms_mps2 | accel_rms_raw_mps2 | id_switch_proxy_per100f | lanes_L10_jit | lanes_R10_jit | lanes_L20_jit | lanes_R20_jit | lanes_L10_pres | lanes_R10_pres | edges_L10_jit | edges_R10_jit | edges_L20_jit | edges_R20_jit | edges_L10_pres | edges_R10_pres |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2d2a | 10.0 | 172 |  |  |  |  |  |  |  |  | 0.0 | 0.0 | 0.0 | 0.009 | 0.005 | 0.018 | 10.5 | 37.2 |
| a2d2b | 10.0 | 172 |  |  |  |  |  |  |  |  | 0.0 | 0.0 | 0.0 | 0.005 | 0.012 | 0.01 | 20.3 | 30.2 |
| kitti_city_cyclist_candidate | 4.0 | 30 |  |  |  |  |  |  |  |  | 0.0 | 0.0 | 0.015 | 0.006 | 0.018 | 0.017 | 93.3 | 93.3 |
| kitti_highway_exit_approach | 4.0 | 40 |  |  |  |  |  | 0.0 |  | 0.003 | 0.0 | 65.0 | 0.008 | 0.007 | 0.011 | 0.013 | 95.0 | 87.5 |
| kitti_highway_final | 4.0 | 30 |  |  |  |  |  |  |  |  | 0.0 | 0.0 | 0.008 | 0.006 | 0.01 | 0.008 | 86.7 | 83.3 |
| kitti_highway_late | 4.0 | 40 |  |  |  |  |  |  |  |  | 0.0 | 0.0 | 0.085 | 0.067 | 0.039 | 0.093 | 95.0 | 95.0 |
| kitti_highway_sample | 4.0 | 40 |  |  |  |  |  | 0.001 |  |  | 0.0 | 55.0 | 0.007 | 0.005 | 0.007 | 0.004 | 95.0 | 92.5 |
| kitti_lane_split_candidate | 4.0 | 30 |  |  |  |  |  |  |  |  | 0.0 | 0.0 | 0.072 | 0.0 | 0.014 | 0.0 | 10.0 | 73.3 |
| munich | 30.0 | 900 |  |  |  |  |  |  |  |  | 0.0 | 0.0 | 0.002 | 0.008 | 0.003 | 0.018 | 55.9 | 73.7 |

coefficient jitter (motion-compensated, median per frame): lanes / edges

- a2d2a: lanes {'dc0_m': None, 'dc1x10_m': None, 'dc2x100_m': None, 'n': 0}; edges {'dc0_m': 0.054, 'dc1x10_m': 0.063, 'dc2x100_m': 0.018, 'n': 269}
- a2d2b: lanes {'dc0_m': None, 'dc1x10_m': None, 'dc2x100_m': None, 'n': 0}; edges {'dc0_m': 0.06, 'dc1x10_m': 0.049, 'dc2x100_m': 0.008, 'n': 275}
- kitti_city_cyclist_candidate: lanes {'dc0_m': None, 'dc1x10_m': None, 'dc2x100_m': None, 'n': 0}; edges {'dc0_m': 0.018, 'dc1x10_m': 0.027, 'dc2x100_m': 0.006, 'n': 54}
- kitti_highway_exit_approach: lanes {'dc0_m': 0.001, 'dc1x10_m': 0.005, 'dc2x100_m': 0.005, 'n': 25}; edges {'dc0_m': 0.009, 'dc1x10_m': 0.019, 'dc2x100_m': 0.006, 'n': 71}
- kitti_highway_final: lanes {'dc0_m': None, 'dc1x10_m': None, 'dc2x100_m': None, 'n': 0}; edges {'dc0_m': 0.019, 'dc1x10_m': 0.015, 'dc2x100_m': 0.003, 'n': 49}
- kitti_highway_late: lanes {'dc0_m': None, 'dc1x10_m': None, 'dc2x100_m': None, 'n': 0}; edges {'dc0_m': 0.056, 'dc1x10_m': 0.014, 'dc2x100_m': 0.005, 'n': 74}
- kitti_highway_sample: lanes {'dc0_m': 0.001, 'dc1x10_m': 0.001, 'dc2x100_m': 0.0, 'n': 21}; edges {'dc0_m': 0.016, 'dc1x10_m': 0.015, 'dc2x100_m': 0.003, 'n': 73}
- kitti_lane_split_candidate: lanes {'dc0_m': None, 'dc1x10_m': None, 'dc2x100_m': None, 'n': 0}; edges {'dc0_m': 0.0, 'dc1x10_m': 0.0, 'dc2x100_m': 0.0, 'n': 31}
- munich: lanes {'dc0_m': None, 'dc1x10_m': None, 'dc2x100_m': None, 'n': 0}; edges {'dc0_m': 0.007, 'dc1x10_m': 0.018, 'dc2x100_m': 0.005, 'n': 1174}
