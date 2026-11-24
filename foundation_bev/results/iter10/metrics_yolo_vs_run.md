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
| a2d2a | 10.0 | 172 | 23 | 0.495 | 157.67 | 0.0 | 0.048 | 0.021 | 0.009 | 0.025 | 15.1 | 26.7 | 0.069 | 0.076 | 0.079 | 0.065 | 95.9 | 94.8 | 0.008 | 0.007 |  | 15.1 | 70.3 | 0.005 | 0.009 | 0.036 | 0.03 | 0.065 |
| a2d2b | 10.0 | 172 | 11 | 0.821 | 93.51 | 0.0 | 0.025 | 0.024 | 0.02 | 0.049 | 8.1 | 2.3 | 0.062 | 0.075 | 0.069 | 0.093 | 97.1 | 97.1 | 0.012 | 0.035 |  | 9.9 | 11.0 | 0.011 | 0.005 | 0.038 | 0.032 | 0.057 |
| kitti_city_cyclist_candidate | 10.0 | 30 | 5 | 0.548 | 153.61 | 0.0 | 0.012 | 0.014 | 0.02 | 0.05 | 100.0 | 100.0 | 0.092 | 0.019 | 0.052 | 0.039 | 93.3 | 93.3 | 0.002 | 0.003 | 0.004 | 100.0 | 100.0 | 0.001 | 0.003 | 0.021 | 0.01 | 0.054 |
| kitti_highway_exit_approach | 10.0 | 40 | 2 | 0.546 | 275.954 | 0.0 | 0.013 | 0.036 | 0.011 | 0.037 | 100.0 | 100.0 | 0.035 | 0.062 | 0.018 | 0.058 | 95.0 | 95.0 | 0.005 | 0.014 | 0.017 | 100.0 | 100.0 | 0.005 | 0.012 | 0.016 | 0.024 | 0.048 |
| kitti_highway_final | 10.0 | 30 | 3 | 0.574 | 370.17 | 0.0 | 0.027 | 0.019 | 0.086 | 0.01 | 93.3 | 90.0 | 0.056 | 0.038 | 0.058 | 0.028 | 90.0 | 86.7 | 0.016 | 0.004 | 0.012 | 66.7 | 90.0 | 0.016 | 0.004 | 0.029 | 0.025 | 0.023 |
| kitti_highway_late | 10.0 | 40 | 5 | 0.534 | 365.63 | 0.0 | 0.086 | 0.05 | 0.066 | 0.07 | 97.5 | 100.0 | 0.105 | 0.109 | 0.092 | 0.102 | 95.0 | 87.5 | 0.011 | 0.007 | 0.019 | 95.0 | 100.0 | 0.011 | 0.01 | 0.02 | 0.015 | 0.035 |
| kitti_highway_sample | 10.0 | 40 | 1 | 0.095 | 509.386 | 0.0 | 0.006 | 0.006 | 0.006 | 0.007 | 100.0 | 100.0 | 0.04 | 0.02 | 0.035 | 0.019 | 95.0 | 95.0 | 0.004 | 0.003 | 0.004 | 100.0 | 100.0 | 0.003 | 0.002 | 0.02 | 0.009 | 0.019 |
| kitti_lane_split_candidate | 10.0 | 30 | 13 | 0.951 | 275.84 | 0.0 | 0.047 | 0.159 | 0.051 | 0.038 | 100.0 | 73.3 | 0.237 | 0.043 | 0.363 | 0.111 | 93.3 | 93.3 | 0.011 | 0.022 | 0.131 | 100.0 | 46.7 | 0.014 | 0.033 | 0.042 | 0.042 | 0.127 |
| munich | 30.0 | 900 | 87 | 0.655 | 1891.94 | 0.0 | 0.01 | 0.007 | 0.008 | 0.007 | 85.9 | 65.4 | 0.026 | 0.062 | 0.018 | 0.054 | 99.8 | 86.2 | 0.003 | 0.002 | 0.006 | 92.6 | 84.8 | 0.002 | 0.002 | 0.01 | 0.022 | 0.04 |

coefficient jitter (motion-compensated, median per frame): lanes / edges

- a2d2a: lanes {'dc0_m': 0.039, 'dc1x10_m': 0.002, 'dc2x100_m': 0.0, 'n': 184}; edges {'dc0_m': 0.158, 'dc1x10_m': 0.126, 'dc2x100_m': 0.019, 'n': 324}
- a2d2b: lanes {'dc0_m': 0.056, 'dc1x10_m': 0.008, 'dc2x100_m': 0.001, 'n': 362}; edges {'dc0_m': 0.166, 'dc1x10_m': 0.139, 'dc2x100_m': 0.027, 'n': 342}
- kitti_city_cyclist_candidate: lanes {'dc0_m': 0.023, 'dc1x10_m': 0.028, 'dc2x100_m': 0.005, 'n': 122}; edges {'dc0_m': 0.119, 'dc1x10_m': 0.11, 'dc2x100_m': 0.024, 'n': 54}
- kitti_highway_exit_approach: lanes {'dc0_m': 0.026, 'dc1x10_m': 0.021, 'dc2x100_m': 0.004, 'n': 78}; edges {'dc0_m': 0.091, 'dc1x10_m': 0.034, 'dc2x100_m': 0.006, 'n': 74}
- kitti_highway_final: lanes {'dc0_m': 0.037, 'dc1x10_m': 0.026, 'dc2x100_m': 0.0, 'n': 50}; edges {'dc0_m': 0.065, 'dc1x10_m': 0.032, 'dc2x100_m': 0.004, 'n': 51}
- kitti_highway_late: lanes {'dc0_m': 0.054, 'dc1x10_m': 0.029, 'dc2x100_m': 0.008, 'n': 74}; edges {'dc0_m': 0.138, 'dc1x10_m': 0.058, 'dc2x100_m': 0.015, 'n': 70}
- kitti_highway_sample: lanes {'dc0_m': 0.008, 'dc1x10_m': 0.007, 'dc2x100_m': 0.001, 'n': 78}; edges {'dc0_m': 0.034, 'dc1x10_m': 0.017, 'dc2x100_m': 0.004, 'n': 74}
- kitti_lane_split_candidate: lanes {'dc0_m': 0.086, 'dc1x10_m': 0.0, 'dc2x100_m': 0.0, 'n': 103}; edges {'dc0_m': 0.379, 'dc1x10_m': 0.269, 'dc2x100_m': 0.03, 'n': 52}
- munich: lanes {'dc0_m': 0.031, 'dc1x10_m': 0.018, 'dc2x100_m': 0.002, 'n': 2874}; edges {'dc0_m': 0.121, 'dc1x10_m': 0.076, 'dc2x100_m': 0.014, 'n': 1764}
