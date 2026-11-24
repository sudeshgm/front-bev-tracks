# Smoothness metrics

accel = RMS BEV acceleration of drawn track positions (m/s^2); idsw = probable ID switches /100 frames;
jit = median |frame-to-frame change| of nearest left/right line lateral offset at z (m); pres = % frames present.
jit2 = median |2nd difference| (lane-change aware: steady sideways drift is not jitter); wjit = median |change of half the L-R spacing| (common mode removed). z=12 columns: Munich nearest ground is ~10.4 m.

## iter3

| clip | fps | frames | n_tracks | accel_rms_mps2 | accel_rms_raw_mps2 | id_switch_proxy_per100f | lanes_L10_jit | lanes_R10_jit | lanes_L20_jit | lanes_R20_jit | lanes_L10_pres | lanes_R10_pres | edges_L10_jit | edges_R10_jit | edges_L20_jit | edges_R20_jit | edges_L10_pres | edges_R10_pres | lanes_L10_jit2 | lanes_R10_jit2 | lanes_10_wjit | lanes_L12_pres | lanes_R12_pres | lanes_L12_jit2 | lanes_R12_jit2 | edges_L10_jit2 | edges_R10_jit2 | edges_10_wjit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2d2a | 10.0 | 172 | 23 | 0.82 | 167.363 | 0.0 | 0.0 | 0.002 | 0.003 | 0.004 | 36.0 | 78.5 |  |  |  | 0.003 | 0.0 | 0.0 | 0.0 | 0.001 | 0.001 | 37.8 | 87.8 | 0.0 | 0.001 |  |  |  |
| a2d2b | 10.0 | 172 | 11 | 0.89 | 93.51 | 0.0 | 0.008 | 0.001 | 0.005 | 0.017 | 3.5 | 14.5 |  |  |  | 0.006 | 0.0 | 0.0 | 0.008 | 0.0 |  | 51.7 | 52.3 | 0.001 | 0.002 |  |  |  |
| kitti_city_cyclist_candidate | 4.0 | 30 | 5 | 0.585 | 24.578 | 0.0 | 0.012 | 0.012 | 0.021 | 0.024 | 93.3 | 93.3 | 0.0 | 0.085 | 0.0 | 0.089 | 43.3 | 93.3 | 0.001 | 0.002 | 0.002 | 93.3 | 93.3 | 0.001 | 0.003 | 0.0 | 0.004 | 0.039 |
| kitti_highway_exit_approach | 4.0 | 40 | 2 | 0.679 | 44.153 | 0.0 | 0.009 | 0.007 | 0.013 | 0.008 | 95.0 | 95.0 | 0.005 | 0.006 | 0.008 | 0.008 | 95.0 | 95.0 | 0.003 | 0.001 | 0.003 | 95.0 | 95.0 | 0.002 | 0.001 | 0.003 | 0.002 | 0.005 |
| kitti_highway_final | 4.0 | 30 | 2 | 0.915 | 49.26 | 0.0 | 0.014 | 0.008 | 0.016 | 0.006 | 90.0 | 90.0 | 0.014 | 0.004 | 0.014 | 0.015 | 80.0 | 86.7 | 0.002 | 0.002 | 0.004 | 90.0 | 90.0 | 0.003 | 0.002 | 0.003 | 0.002 | 0.006 |
| kitti_highway_late | 4.0 | 40 | 5 | 1.827 | 57.608 | 0.0 | 0.041 | 0.052 | 0.047 | 0.039 | 95.0 | 95.0 | 0.044 | 0.058 | 0.023 | 0.058 | 90.0 | 95.0 | 0.009 | 0.007 | 0.024 | 95.0 | 95.0 | 0.009 | 0.008 | 0.005 | 0.006 | 0.016 |
| kitti_highway_sample | 4.0 | 40 | 2 | 0.584 | 70.321 | 0.0 | 0.005 | 0.003 | 0.005 | 0.005 | 95.0 | 95.0 | 0.009 | 0.004 | 0.006 | 0.006 | 85.0 | 95.0 | 0.001 | 0.001 | 0.002 | 95.0 | 95.0 | 0.001 | 0.001 | 0.006 | 0.001 | 0.005 |
| kitti_lane_split_candidate | 4.0 | 30 | 15 | 1.134 | 34.952 | 0.0 |  | 0.003 |  | 0.004 | 0.0 | 40.0 |  | 0.002 |  |  | 0.0 | 73.3 |  | 0.002 |  | 0.0 | 36.7 |  | 0.001 |  | 0.0 |  |
| munich | 30.0 | 900 | 87 | 0.911 | 1875.02 | 0.0 | 0.003 | 0.005 | 0.004 | 0.006 | 92.2 | 97.6 | 0.005 |  | 0.007 | 0.0 | 99.8 | 0.0 | 0.001 | 0.001 | 0.003 | 96.1 | 99.1 | 0.001 | 0.001 | 0.001 |  |  |

coefficient jitter (motion-compensated, median per frame): lanes / edges

- a2d2a: lanes {'dc0_m': 0.005, 'dc1x10_m': 0.006, 'dc2x100_m': 0.0, 'n': 314}; edges {'dc0_m': 0.003, 'dc1x10_m': 0.005, 'dc2x100_m': 0.001, 'n': 137}
- a2d2b: lanes {'dc0_m': 0.013, 'dc1x10_m': 0.011, 'dc2x100_m': 0.002, 'n': 495}; edges {'dc0_m': 0.038, 'dc1x10_m': 0.028, 'dc2x100_m': 0.005, 'n': 91}
- kitti_city_cyclist_candidate: lanes {'dc0_m': 0.011, 'dc1x10_m': 0.011, 'dc2x100_m': 0.004, 'n': 108}; edges {'dc0_m': 0.023, 'dc1x10_m': 0.134, 'dc2x100_m': 0.05, 'n': 47}
- kitti_highway_exit_approach: lanes {'dc0_m': 0.009, 'dc1x10_m': 0.006, 'dc2x100_m': 0.003, 'n': 93}; edges {'dc0_m': 0.015, 'dc1x10_m': 0.025, 'dc2x100_m': 0.005, 'n': 71}
- kitti_highway_final: lanes {'dc0_m': 0.008, 'dc1x10_m': 0.005, 'dc2x100_m': 0.002, 'n': 63}; edges {'dc0_m': 0.024, 'dc1x10_m': 0.029, 'dc2x100_m': 0.005, 'n': 48}
- kitti_highway_late: lanes {'dc0_m': 0.044, 'dc1x10_m': 0.013, 'dc2x100_m': 0.009, 'n': 111}; edges {'dc0_m': 0.039, 'dc1x10_m': 0.032, 'dc2x100_m': 0.007, 'n': 72}
- kitti_highway_sample: lanes {'dc0_m': 0.006, 'dc1x10_m': 0.003, 'dc2x100_m': 0.002, 'n': 103}; edges {'dc0_m': 0.008, 'dc1x10_m': 0.014, 'dc2x100_m': 0.005, 'n': 70}
- kitti_lane_split_candidate: lanes {'dc0_m': 0.0, 'dc1x10_m': 0.0, 'dc2x100_m': 0.0, 'n': 26}; edges {'dc0_m': 0.002, 'dc1x10_m': 0.005, 'dc2x100_m': 0.006, 'n': 21}
- munich: lanes {'dc0_m': 0.013, 'dc1x10_m': 0.016, 'dc2x100_m': 0.004, 'n': 2687}; edges {'dc0_m': 0.014, 'dc1x10_m': 0.016, 'dc2x100_m': 0.004, 'n': 976}
