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
| a2d2a | 10.0 | 172 | 22 | 0.901 | 132.302 | 0.0 | 0.041 | 0.018 | 0.011 | 0.032 | 11.0 | 25.0 | 0.105 | 0.116 | 0.079 | 0.093 | 76.2 | 71.5 | 0.001 | 0.011 |  | 12.8 | 66.9 | 0.001 | 0.008 | 0.064 | 0.066 | 0.087 |
| a2d2b | 10.0 | 172 | 11 | 0.827 | 97.821 | 0.0 | 0.043 | 0.025 | 0.027 | 0.058 | 8.7 | 1.7 | 0.082 | 0.091 | 0.068 | 0.103 | 85.5 | 83.7 | 0.014 | 0.05 |  | 9.3 | 9.9 | 0.012 | 0.021 | 0.055 | 0.052 | 0.075 |
| kitti_city_cyclist_candidate | 10.0 | 30 | 5 | 0.566 | 153.61 | 0.0 | 0.01 | 0.015 | 0.021 | 0.05 | 93.3 | 93.3 | 0.117 | 0.022 | 0.056 | 0.051 | 93.3 | 93.3 | 0.004 | 0.01 | 0.008 | 93.3 | 93.3 | 0.006 | 0.016 | 0.065 | 0.026 | 0.07 |
| kitti_highway_exit_approach | 10.0 | 40 | 2 | 0.546 | 275.954 | 0.0 | 0.011 | 0.034 | 0.013 | 0.053 | 95.0 | 95.0 | 0.038 | 0.072 | 0.024 | 0.072 | 95.0 | 95.0 | 0.007 | 0.018 | 0.033 | 95.0 | 95.0 | 0.007 | 0.018 | 0.019 | 0.032 | 0.055 |
| kitti_highway_final | 10.0 | 30 | 3 | 0.638 | 386.621 | 0.0 | 0.03 | 0.017 | 0.034 | 0.016 | 93.3 | 90.0 | 0.066 | 0.058 | 0.076 | 0.047 | 90.0 | 86.7 | 0.032 | 0.006 | 0.019 | 93.3 | 90.0 | 0.027 | 0.006 | 0.034 | 0.04 | 0.034 |
| kitti_highway_late | 10.0 | 40 | 5 | 0.575 | 354.156 | 0.0 | 0.079 | 0.059 | 0.055 | 0.093 | 92.5 | 95.0 | 0.072 | 0.089 | 0.085 | 0.098 | 95.0 | 77.5 | 0.019 | 0.016 | 0.02 | 92.5 | 95.0 | 0.019 | 0.021 | 0.026 | 0.034 | 0.048 |
| kitti_highway_sample | 10.0 | 40 | 1 | 0.095 | 509.386 | 0.0 | 0.011 | 0.005 | 0.014 | 0.007 | 95.0 | 95.0 | 0.053 | 0.024 | 0.049 | 0.026 | 95.0 | 95.0 | 0.007 | 0.005 | 0.007 | 95.0 | 95.0 | 0.008 | 0.004 | 0.027 | 0.017 | 0.026 |
| kitti_lane_split_candidate | 10.0 | 30 | 13 | 0.997 | 274.066 | 0.0 | 0.134 | 0.145 | 0.138 | 0.054 | 86.7 | 66.7 | 0.143 | 0.073 | 0.347 | 0.258 | 40.0 | 83.3 | 0.047 | 0.025 | 0.066 | 86.7 | 60.0 | 0.047 | 0.083 | 0.049 | 0.067 | 0.083 |
| munich | 30.0 | 900 | 87 | 0.912 | 1876.093 | 0.0 | 0.018 | 0.007 | 0.017 | 0.009 | 85.7 | 65.3 | 0.056 | 0.079 | 0.054 | 0.059 | 87.9 | 64.0 | 0.009 | 0.003 | 0.01 | 93.1 | 84.7 | 0.008 | 0.003 | 0.033 | 0.037 | 0.053 |

coefficient jitter (motion-compensated, median per frame): lanes / edges

- a2d2a: lanes {'dc0_m': 0.043, 'dc1x10_m': 0.006, 'dc2x100_m': 0.0, 'n': 185}; edges {'dc0_m': 0.125, 'dc1x10_m': 0.106, 'dc2x100_m': 0.024, 'n': 293}
- a2d2b: lanes {'dc0_m': 0.053, 'dc1x10_m': 0.006, 'dc2x100_m': 0.001, 'n': 459}; edges {'dc0_m': 0.139, 'dc1x10_m': 0.114, 'dc2x100_m': 0.022, 'n': 277}
- kitti_city_cyclist_candidate: lanes {'dc0_m': 0.028, 'dc1x10_m': 0.033, 'dc2x100_m': 0.006, 'n': 121}; edges {'dc0_m': 0.119, 'dc1x10_m': 0.11, 'dc2x100_m': 0.024, 'n': 54}
- kitti_highway_exit_approach: lanes {'dc0_m': 0.038, 'dc1x10_m': 0.025, 'dc2x100_m': 0.004, 'n': 110}; edges {'dc0_m': 0.091, 'dc1x10_m': 0.034, 'dc2x100_m': 0.006, 'n': 74}
- kitti_highway_final: lanes {'dc0_m': 0.041, 'dc1x10_m': 0.032, 'dc2x100_m': 0.006, 'n': 94}; edges {'dc0_m': 0.062, 'dc1x10_m': 0.032, 'dc2x100_m': 0.004, 'n': 51}
- kitti_highway_late: lanes {'dc0_m': 0.044, 'dc1x10_m': 0.03, 'dc2x100_m': 0.008, 'n': 102}; edges {'dc0_m': 0.133, 'dc1x10_m': 0.058, 'dc2x100_m': 0.016, 'n': 66}
- kitti_highway_sample: lanes {'dc0_m': 0.01, 'dc1x10_m': 0.008, 'dc2x100_m': 0.002, 'n': 101}; edges {'dc0_m': 0.034, 'dc1x10_m': 0.017, 'dc2x100_m': 0.004, 'n': 74}
- kitti_lane_split_candidate: lanes {'dc0_m': 0.112, 'dc1x10_m': 0.0, 'dc2x100_m': 0.0, 'n': 106}; edges {'dc0_m': 0.23, 'dc1x10_m': 0.268, 'dc2x100_m': 0.034, 'n': 34}
- munich: lanes {'dc0_m': 0.033, 'dc1x10_m': 0.017, 'dc2x100_m': 0.002, 'n': 3103}; edges {'dc0_m': 0.092, 'dc1x10_m': 0.06, 'dc2x100_m': 0.011, 'n': 1382}
