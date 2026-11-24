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
| a2d2a | 10.0 | 172 | 23 | 0.791 | 157.67 | 0.0 | 0.047 | 0.026 | 0.013 | 0.029 | 12.8 | 22.7 | 0.103 | 0.115 | 0.068 | 0.084 | 79.7 | 77.3 | 0.001 | 0.009 |  | 12.8 | 57.6 | 0.001 | 0.006 | 0.067 | 0.06 | 0.086 |
| a2d2b | 10.0 | 172 | 11 | 0.89 | 93.51 | 0.0 | 0.044 | 0.025 | 0.029 | 0.057 | 8.7 | 1.7 | 0.088 | 0.091 | 0.09 | 0.122 | 97.1 | 97.1 | 0.017 | 0.05 |  | 8.1 | 11.6 | 0.013 | 0.019 | 0.059 | 0.056 | 0.086 |
| kitti_city_cyclist_candidate | 10.0 | 30 | 5 | 0.566 | 153.61 | 0.0 | 0.011 | 0.015 | 0.021 | 0.058 | 90.0 | 90.0 | 0.116 | 0.022 | 0.055 | 0.051 | 93.3 | 93.3 | 0.004 | 0.011 | 0.009 | 90.0 | 90.0 | 0.006 | 0.016 | 0.064 | 0.026 | 0.069 |
| kitti_highway_exit_approach | 10.0 | 40 | 2 | 0.546 | 275.954 | 0.0 | 0.012 | 0.036 | 0.014 | 0.053 | 92.5 | 92.5 | 0.038 | 0.072 | 0.024 | 0.072 | 95.0 | 95.0 | 0.007 | 0.019 | 0.034 | 92.5 | 92.5 | 0.008 | 0.019 | 0.019 | 0.032 | 0.055 |
| kitti_highway_final | 10.0 | 30 | 3 | 0.626 | 370.17 | 0.0 | 0.035 | 0.015 | 0.042 | 0.016 | 83.3 | 86.7 | 0.064 | 0.058 | 0.07 | 0.047 | 90.0 | 86.7 | 0.033 | 0.006 | 0.019 | 83.3 | 86.7 | 0.027 | 0.006 | 0.033 | 0.04 | 0.034 |
| kitti_highway_late | 10.0 | 40 | 5 | 0.575 | 365.63 | 0.0 | 0.088 | 0.074 | 0.063 | 0.097 | 87.5 | 92.5 | 0.074 | 0.111 | 0.092 | 0.111 | 95.0 | 87.5 | 0.018 | 0.019 | 0.022 | 87.5 | 92.5 | 0.016 | 0.029 | 0.023 | 0.04 | 0.052 |
| kitti_highway_sample | 10.0 | 40 | 1 | 0.095 | 509.386 | 0.0 | 0.011 | 0.005 | 0.013 | 0.007 | 92.5 | 92.5 | 0.053 | 0.024 | 0.049 | 0.026 | 95.0 | 95.0 | 0.007 | 0.005 | 0.007 | 92.5 | 92.5 | 0.008 | 0.004 | 0.027 | 0.017 | 0.026 |
| kitti_lane_split_candidate | 10.0 | 30 | 13 | 1.074 | 275.84 | 0.0 | 0.052 | 0.13 | 0.054 | 0.05 | 90.0 | 46.7 | 0.213 | 0.072 | 0.327 | 0.274 | 93.3 | 93.3 | 0.062 | 0.021 | 0.164 | 90.0 | 40.0 | 0.061 | 0.404 | 0.069 | 0.071 | 0.099 |
| munich | 30.0 | 900 | 87 | 0.911 | 1875.056 | 0.0 | 0.018 | 0.008 | 0.017 | 0.009 | 84.9 | 56.9 | 0.061 | 0.081 | 0.057 | 0.069 | 99.2 | 92.4 | 0.009 | 0.004 | 0.01 | 92.8 | 74.0 | 0.008 | 0.004 | 0.034 | 0.043 | 0.053 |

coefficient jitter (motion-compensated, median per frame): lanes / edges

- a2d2a: lanes {'dc0_m': 0.029, 'dc1x10_m': 0.003, 'dc2x100_m': 0.0, 'n': 135}; edges {'dc0_m': 0.119, 'dc1x10_m': 0.103, 'dc2x100_m': 0.021, 'n': 324}
- a2d2b: lanes {'dc0_m': 0.053, 'dc1x10_m': 0.009, 'dc2x100_m': 0.001, 'n': 374}; edges {'dc0_m': 0.166, 'dc1x10_m': 0.136, 'dc2x100_m': 0.026, 'n': 341}
- kitti_city_cyclist_candidate: lanes {'dc0_m': 0.026, 'dc1x10_m': 0.03, 'dc2x100_m': 0.005, 'n': 114}; edges {'dc0_m': 0.118, 'dc1x10_m': 0.11, 'dc2x100_m': 0.024, 'n': 54}
- kitti_highway_exit_approach: lanes {'dc0_m': 0.039, 'dc1x10_m': 0.025, 'dc2x100_m': 0.004, 'n': 106}; edges {'dc0_m': 0.091, 'dc1x10_m': 0.034, 'dc2x100_m': 0.006, 'n': 74}
- kitti_highway_final: lanes {'dc0_m': 0.041, 'dc1x10_m': 0.034, 'dc2x100_m': 0.006, 'n': 84}; edges {'dc0_m': 0.065, 'dc1x10_m': 0.032, 'dc2x100_m': 0.004, 'n': 51}
- kitti_highway_late: lanes {'dc0_m': 0.049, 'dc1x10_m': 0.024, 'dc2x100_m': 0.008, 'n': 98}; edges {'dc0_m': 0.138, 'dc1x10_m': 0.058, 'dc2x100_m': 0.015, 'n': 70}
- kitti_highway_sample: lanes {'dc0_m': 0.01, 'dc1x10_m': 0.007, 'dc2x100_m': 0.002, 'n': 97}; edges {'dc0_m': 0.034, 'dc1x10_m': 0.017, 'dc2x100_m': 0.004, 'n': 74}
- kitti_lane_split_candidate: lanes {'dc0_m': 0.073, 'dc1x10_m': 0.0, 'dc2x100_m': 0.0, 'n': 82}; edges {'dc0_m': 0.276, 'dc1x10_m': 0.258, 'dc2x100_m': 0.035, 'n': 52}
- munich: lanes {'dc0_m': 0.034, 'dc1x10_m': 0.019, 'dc2x100_m': 0.003, 'n': 2759}; edges {'dc0_m': 0.103, 'dc1x10_m': 0.066, 'dc2x100_m': 0.012, 'n': 1760}
