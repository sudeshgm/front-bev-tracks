# Smoothness metrics

accel = RMS BEV acceleration of drawn track positions (m/s^2); idsw = probable ID switches /100 frames;
jit = median |frame-to-frame change| of nearest left/right line lateral offset at z (m); pres = % frames present.
jit2 = median |2nd difference| (lane-change aware: steady sideways drift is not jitter); wjit = median |change of half the L-R spacing| (common mode removed). z=12 columns: Munich nearest ground is ~10.4 m.

## yolopv2_baseline

| clip | fps | frames | n_tracks | accel_rms_mps2 | accel_rms_raw_mps2 | id_switch_proxy_per100f | lanes_L10_jit | lanes_R10_jit | lanes_L20_jit | lanes_R20_jit | lanes_L10_pres | lanes_R10_pres | edges_L10_jit | edges_R10_jit | edges_L20_jit | edges_R20_jit | edges_L10_pres | edges_R10_pres | lanes_L10_jit2 | lanes_R10_jit2 | lanes_10_wjit | lanes_L12_pres | lanes_R12_pres | lanes_L12_jit2 | lanes_R12_jit2 | edges_L10_jit2 | edges_R10_jit2 | edges_10_wjit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kitti_city_cyclist_candidate | 4.0 | 30 | 3 | 0.491 |  | 0.0 | 0.008 | 0.013 | 0.009 | 0.016 | 100.0 | 100.0 |  |  |  |  | 0.0 | 0.0 | 0.001 | 0.004 | 0.005 | 100.0 | 100.0 | 0.001 | 0.002 |  |  |  |
| kitti_highway_exit_approach | 4.0 | 40 | 0 |  |  |  | 0.011 | 0.012 | 0.011 | 0.012 | 100.0 | 100.0 |  |  |  |  | 0.0 | 0.0 | 0.013 | 0.009 | 0.007 | 100.0 | 100.0 | 0.012 | 0.007 |  |  |  |
| kitti_highway_final | 4.0 | 30 | 3 | 0.755 |  | 0.0 | 0.021 | 0.019 | 0.025 | 0.022 | 100.0 | 100.0 |  |  |  |  | 0.0 | 0.0 | 0.024 | 0.017 | 0.008 | 100.0 | 100.0 | 0.019 | 0.016 |  |  |  |
| kitti_highway_late | 4.0 | 40 | 4 | 0.495 |  | 0.0 | 0.065 | 0.086 | 0.071 | 0.087 | 100.0 | 100.0 |  |  |  |  | 0.0 | 0.0 | 0.014 | 0.014 | 0.009 | 100.0 | 100.0 | 0.013 | 0.014 |  |  |  |
| kitti_highway_sample | 4.0 | 40 | 0 |  |  |  | 0.012 | 0.007 | 0.013 | 0.008 | 100.0 | 100.0 |  |  |  |  | 0.0 | 0.0 | 0.019 | 0.005 | 0.006 | 100.0 | 100.0 | 0.016 | 0.005 |  |  |  |
| kitti_lane_split_candidate | 4.0 | 30 | 10 | 0.615 |  | 0.0 | 0.24 | 0.308 | 0.389 | 0.543 | 76.7 | 26.7 |  |  |  |  | 0.0 | 0.0 | 0.257 | 0.721 | 0.092 | 80.0 | 26.7 | 0.231 | 0.704 |  |  |  |

## working

| clip | fps | frames | n_tracks | accel_rms_mps2 | accel_rms_raw_mps2 | id_switch_proxy_per100f | lanes_L10_jit | lanes_R10_jit | lanes_L20_jit | lanes_R20_jit | lanes_L10_pres | lanes_R10_pres | edges_L10_jit | edges_R10_jit | edges_L20_jit | edges_R20_jit | edges_L10_pres | edges_R10_pres | lanes_L10_jit2 | lanes_R10_jit2 | lanes_10_wjit | lanes_L12_pres | lanes_R12_pres | lanes_L12_jit2 | lanes_R12_jit2 | edges_L10_jit2 | edges_R10_jit2 | edges_10_wjit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2d2a | 10.0 | 172 | 28 | 0.774 | 179.762 | 0.0 | 0.021 | 0.016 | 0.016 | 0.018 | 13.4 | 32.0 | 0.102 | 0.134 | 0.085 | 0.088 | 79.7 | 80.8 | 0.001 | 0.01 |  | 10.5 | 70.9 | 0.001 | 0.006 | 0.097 | 0.119 | 0.1 |
| a2d2b | 10.0 | 172 | 11 | 0.89 | 93.471 | 0.0 | 0.026 | 0.043 | 0.025 | 0.061 | 7.6 | 2.9 | 0.123 | 0.083 | 0.079 | 0.116 | 98.3 | 98.3 | 0.009 | 0.046 |  | 14.0 | 8.1 | 0.013 | 0.046 | 0.081 | 0.057 | 0.073 |
| kitti_city_cyclist_candidate | 10.0 | 30 | 5 | 0.566 | 153.61 | 0.0 | 0.013 | 0.012 | 0.022 | 0.04 | 93.3 | 93.3 | 0.086 | 0.023 | 0.127 | 0.045 | 83.3 | 93.3 | 0.005 | 0.013 | 0.006 | 93.3 | 93.3 | 0.005 | 0.021 | 0.072 | 0.023 | 0.052 |
| kitti_highway_exit_approach | 10.0 | 40 | 2 | 0.541 | 258.438 | 0.0 | 0.012 | 0.02 | 0.021 | 0.024 | 95.0 | 90.0 | 0.027 | 0.058 | 0.035 | 0.062 | 95.0 | 95.0 | 0.01 | 0.006 | 0.014 | 95.0 | 95.0 | 0.006 | 0.007 | 0.021 | 0.039 | 0.049 |
| kitti_highway_final | 10.0 | 30 | 3 | 0.626 | 348.814 | 0.0 | 0.056 | 0.018 | 0.042 | 0.02 | 93.3 | 90.0 | 0.045 | 0.046 | 0.063 | 0.045 | 90.0 | 86.7 | 0.037 | 0.011 | 0.019 | 93.3 | 90.0 | 0.022 | 0.009 | 0.029 | 0.036 | 0.025 |
| kitti_highway_late | 10.0 | 40 | 6 | 0.651 | 262.635 | 0.0 | 0.069 | 0.053 | 0.079 | 0.085 | 92.5 | 95.0 | 0.074 | 0.082 | 0.087 | 0.065 | 95.0 | 87.5 | 0.014 | 0.014 | 0.02 | 92.5 | 95.0 | 0.022 | 0.022 | 0.038 | 0.028 | 0.045 |
| kitti_highway_sample | 10.0 | 40 | 1 | 0.108 | 376.045 | 0.0 | 0.01 | 0.006 | 0.009 | 0.009 | 95.0 | 95.0 | 0.042 | 0.02 | 0.043 | 0.031 | 95.0 | 95.0 | 0.004 | 0.004 | 0.006 | 95.0 | 95.0 | 0.009 | 0.004 | 0.026 | 0.016 | 0.018 |
| kitti_lane_split_candidate | 10.0 | 30 | 13 | 1.052 | 252.774 | 0.0 | 0.037 | 0.212 | 0.035 | 0.179 | 93.3 | 76.7 | 0.222 | 0.206 | 0.317 | 0.351 | 93.3 | 93.3 | 0.043 | 0.001 | 0.108 | 93.3 | 76.7 | 0.036 | 0.026 | 0.085 | 0.128 | 0.15 |
| munich | 30.0 | 900 | 87 | 0.917 | 1874.467 | 0.0 | 0.014 | 0.006 | 0.014 | 0.008 | 87.6 | 65.4 | 0.053 | 0.051 | 0.049 | 0.06 | 99.8 | 98.0 | 0.009 | 0.003 | 0.007 | 93.4 | 85.7 | 0.008 | 0.004 | 0.032 | 0.039 | 0.04 |

coefficient jitter (motion-compensated, median per frame): lanes / edges

- a2d2a: lanes {'dc0_m': 0.027, 'dc1x10_m': 0.007, 'dc2x100_m': 0.0, 'n': 189}; edges {'dc0_m': 0.103, 'dc1x10_m': 0.085, 'dc2x100_m': 0.017, 'n': 347}
- a2d2b: lanes {'dc0_m': 0.059, 'dc1x10_m': 0.012, 'dc2x100_m': 0.001, 'n': 565}; edges {'dc0_m': 0.196, 'dc1x10_m': 0.176, 'dc2x100_m': 0.032, 'n': 370}
- kitti_city_cyclist_candidate: lanes {'dc0_m': 0.044, 'dc1x10_m': 0.042, 'dc2x100_m': 0.009, 'n': 178}; edges {'dc0_m': 0.131, 'dc1x10_m': 0.112, 'dc2x100_m': 0.025, 'n': 52}
- kitti_highway_exit_approach: lanes {'dc0_m': 0.027, 'dc1x10_m': 0.031, 'dc2x100_m': 0.005, 'n': 107}; edges {'dc0_m': 0.085, 'dc1x10_m': 0.041, 'dc2x100_m': 0.007, 'n': 74}
- kitti_highway_final: lanes {'dc0_m': 0.054, 'dc1x10_m': 0.043, 'dc2x100_m': 0.005, 'n': 106}; edges {'dc0_m': 0.056, 'dc1x10_m': 0.038, 'dc2x100_m': 0.006, 'n': 51}
- kitti_highway_late: lanes {'dc0_m': 0.046, 'dc1x10_m': 0.068, 'dc2x100_m': 0.018, 'n': 116}; edges {'dc0_m': 0.127, 'dc1x10_m': 0.073, 'dc2x100_m': 0.017, 'n': 71}
- kitti_highway_sample: lanes {'dc0_m': 0.015, 'dc1x10_m': 0.012, 'dc2x100_m': 0.003, 'n': 105}; edges {'dc0_m': 0.038, 'dc1x10_m': 0.024, 'dc2x100_m': 0.005, 'n': 74}
- kitti_lane_split_candidate: lanes {'dc0_m': 0.082, 'dc1x10_m': 0.0, 'dc2x100_m': 0.0, 'n': 161}; edges {'dc0_m': 0.267, 'dc1x10_m': 0.293, 'dc2x100_m': 0.05, 'n': 59}
- munich: lanes {'dc0_m': 0.044, 'dc1x10_m': 0.021, 'dc2x100_m': 0.002, 'n': 3957}; edges {'dc0_m': 0.107, 'dc1x10_m': 0.077, 'dc2x100_m': 0.014, 'n': 1987}
