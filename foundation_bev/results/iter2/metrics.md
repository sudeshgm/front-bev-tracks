# Smoothness metrics

accel = RMS BEV acceleration of drawn track positions (m/s^2); idsw = probable ID switches /100 frames;
jit = median |frame-to-frame change| of nearest left/right line lateral offset at z (m); pres = % frames present.

## iter2

| clip | fps | frames | n_tracks | accel_rms_mps2 | accel_rms_raw_mps2 | id_switch_proxy_per100f | lanes_L10_jit | lanes_R10_jit | lanes_L20_jit | lanes_R20_jit | lanes_L10_pres | lanes_R10_pres | edges_L10_jit | edges_R10_jit | edges_L20_jit | edges_R20_jit | edges_L10_pres | edges_R10_pres |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2d2a | 10.0 | 172 | 49 | 1.261 | 410.505 | 0.0 | 0.0 | 0.0 | 0.0 | 0.003 | 9.3 | 9.3 |  |  | 0.0 | 0.0 | 0.0 | 0.0 |
| a2d2b | 10.0 | 172 | 76 | 1.119 | 394.327 | 0.0 |  |  | 0.0 | 0.003 | 0.0 | 0.0 |  |  | 0.0 | 0.002 | 0.0 | 0.0 |
| kitti_city_cyclist_candidate | 4.0 | 30 | 10 | 0.474 | 37.088 | 0.0 | 0.01 | 0.007 | 0.016 | 0.017 | 93.3 | 93.3 | 0.0 | 0.085 | 0.0 | 0.089 | 43.3 | 93.3 |
| kitti_highway_exit_approach | 4.0 | 40 | 6 | 0.628 | 56.102 | 0.0 | 0.009 | 0.007 | 0.011 | 0.01 | 95.0 | 95.0 | 0.005 | 0.006 | 0.008 | 0.008 | 95.0 | 95.0 |
| kitti_highway_final | 4.0 | 30 | 9 | 0.838 | 180.557 | 0.0 | 0.014 | 0.011 | 0.017 | 0.006 | 90.0 | 86.7 | 0.014 | 0.004 | 0.014 | 0.015 | 80.0 | 86.7 |
| kitti_highway_late | 4.0 | 40 | 13 | 1.697 | 115.594 | 0.0 | 0.063 | 0.027 | 0.073 | 0.081 | 95.0 | 95.0 | 0.044 | 0.058 | 0.023 | 0.058 | 90.0 | 95.0 |
| kitti_highway_sample | 4.0 | 40 | 5 | 0.504 | 107.467 | 0.0 | 0.004 | 0.003 | 0.005 | 0.004 | 95.0 | 95.0 | 0.009 | 0.004 | 0.006 | 0.005 | 85.0 | 95.0 |
| kitti_lane_split_candidate | 4.0 | 30 | 15 | 1.131 | 29.628 | 0.0 |  | 0.065 |  |  | 0.0 | 70.0 |  |  |  |  | 0.0 | 0.0 |
| munich | 30.0 | 900 | 171 | 1.079 | 2621.939 | 0.0 |  |  | 0.006 | 0.005 | 0.0 | 0.0 | 0.004 |  | 0.006 | 0.0 | 68.2 | 0.0 |

coefficient jitter (motion-compensated, median per frame): lanes / edges

- a2d2a: lanes {'dc0_m': 0.0, 'dc1x10_m': 0.0, 'dc2x100_m': 0.0, 'n': 77}; edges {'dc0_m': 0.0, 'dc1x10_m': 0.0, 'dc2x100_m': 0.0, 'n': 139}
- a2d2b: lanes {'dc0_m': 0.001, 'dc1x10_m': 0.001, 'dc2x100_m': 0.0, 'n': 335}; edges {'dc0_m': 0.0, 'dc1x10_m': 0.0, 'dc2x100_m': 0.0, 'n': 119}
- kitti_city_cyclist_candidate: lanes {'dc0_m': 0.013, 'dc1x10_m': 0.02, 'dc2x100_m': 0.005, 'n': 108}; edges {'dc0_m': 0.023, 'dc1x10_m': 0.134, 'dc2x100_m': 0.05, 'n': 47}
- kitti_highway_exit_approach: lanes {'dc0_m': 0.01, 'dc1x10_m': 0.012, 'dc2x100_m': 0.003, 'n': 94}; edges {'dc0_m': 0.015, 'dc1x10_m': 0.025, 'dc2x100_m': 0.005, 'n': 71}
- kitti_highway_final: lanes {'dc0_m': 0.015, 'dc1x10_m': 0.013, 'dc2x100_m': 0.002, 'n': 66}; edges {'dc0_m': 0.024, 'dc1x10_m': 0.029, 'dc2x100_m': 0.005, 'n': 48}
- kitti_highway_late: lanes {'dc0_m': 0.026, 'dc1x10_m': 0.018, 'dc2x100_m': 0.004, 'n': 111}; edges {'dc0_m': 0.039, 'dc1x10_m': 0.032, 'dc2x100_m': 0.007, 'n': 72}
- kitti_highway_sample: lanes {'dc0_m': 0.005, 'dc1x10_m': 0.004, 'dc2x100_m': 0.001, 'n': 103}; edges {'dc0_m': 0.008, 'dc1x10_m': 0.014, 'dc2x100_m': 0.005, 'n': 70}
- kitti_lane_split_candidate: lanes {'dc0_m': 0.001, 'dc1x10_m': 0.007, 'dc2x100_m': 0.0, 'n': 30}; edges {'dc0_m': None, 'dc1x10_m': None, 'dc2x100_m': None, 'n': 0}
- munich: lanes {'dc0_m': 0.018, 'dc1x10_m': 0.015, 'dc2x100_m': 0.003, 'n': 2323}; edges {'dc0_m': 0.013, 'dc1x10_m': 0.014, 'dc2x100_m': 0.004, 'n': 974}
