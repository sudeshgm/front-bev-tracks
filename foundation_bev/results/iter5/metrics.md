# Smoothness metrics

accel = RMS BEV acceleration of drawn track positions (m/s^2); idsw = probable ID switches /100 frames;
jit = median |frame-to-frame change| of nearest left/right line lateral offset at z (m); pres = % frames present.
jit2 = median |2nd difference| (lane-change aware: steady sideways drift is not jitter); wjit = median |change of half the L-R spacing| (common mode removed). z=12 columns: Munich nearest ground is ~10.4 m.

## iter5

| clip | fps | frames | n_tracks | accel_rms_mps2 | accel_rms_raw_mps2 | id_switch_proxy_per100f | lanes_L10_jit | lanes_R10_jit | lanes_L20_jit | lanes_R20_jit | lanes_L10_pres | lanes_R10_pres | edges_L10_jit | edges_R10_jit | edges_L20_jit | edges_R20_jit | edges_L10_pres | edges_R10_pres | lanes_L10_jit2 | lanes_R10_jit2 | lanes_10_wjit | lanes_L12_pres | lanes_R12_pres | lanes_L12_jit2 | lanes_R12_jit2 | edges_L10_jit2 | edges_R10_jit2 | edges_10_wjit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kitti_highway_exit_approach | 10.0 | 40 | 2 | 0.546 | 275.954 | 0.0 | 0.012 | 0.036 | 0.014 | 0.053 | 92.5 | 92.5 | 0.038 | 0.072 | 0.024 | 0.072 | 95.0 | 95.0 | 0.007 | 0.019 | 0.034 | 92.5 | 92.5 | 0.008 | 0.019 | 0.019 | 0.032 | 0.055 |
| kitti_highway_final | 10.0 | 30 | 3 | 0.626 | 370.17 | 0.0 | 0.035 | 0.015 | 0.042 | 0.016 | 83.3 | 86.7 | 0.064 | 0.058 | 0.07 | 0.047 | 90.0 | 86.7 | 0.033 | 0.006 | 0.019 | 83.3 | 86.7 | 0.027 | 0.006 | 0.033 | 0.04 | 0.034 |
| kitti_highway_late | 10.0 | 40 | 5 | 0.575 | 365.63 | 0.0 | 0.088 | 0.074 | 0.063 | 0.097 | 87.5 | 92.5 | 0.074 | 0.111 | 0.092 | 0.111 | 95.0 | 87.5 | 0.018 | 0.019 | 0.022 | 87.5 | 92.5 | 0.016 | 0.029 | 0.023 | 0.04 | 0.052 |
| kitti_highway_sample | 10.0 | 40 | 1 | 0.095 | 509.386 | 0.0 | 0.011 | 0.005 | 0.013 | 0.007 | 92.5 | 92.5 | 0.053 | 0.024 | 0.049 | 0.026 | 95.0 | 95.0 | 0.007 | 0.005 | 0.007 | 92.5 | 92.5 | 0.008 | 0.004 | 0.027 | 0.017 | 0.026 |

coefficient jitter (motion-compensated, median per frame): lanes / edges

- kitti_highway_exit_approach: lanes {'dc0_m': 0.039, 'dc1x10_m': 0.025, 'dc2x100_m': 0.004, 'n': 106}; edges {'dc0_m': 0.091, 'dc1x10_m': 0.034, 'dc2x100_m': 0.006, 'n': 74}
- kitti_highway_final: lanes {'dc0_m': 0.041, 'dc1x10_m': 0.034, 'dc2x100_m': 0.006, 'n': 84}; edges {'dc0_m': 0.065, 'dc1x10_m': 0.032, 'dc2x100_m': 0.004, 'n': 51}
- kitti_highway_late: lanes {'dc0_m': 0.049, 'dc1x10_m': 0.024, 'dc2x100_m': 0.008, 'n': 98}; edges {'dc0_m': 0.138, 'dc1x10_m': 0.058, 'dc2x100_m': 0.015, 'n': 70}
- kitti_highway_sample: lanes {'dc0_m': 0.01, 'dc1x10_m': 0.007, 'dc2x100_m': 0.002, 'n': 97}; edges {'dc0_m': 0.034, 'dc1x10_m': 0.017, 'dc2x100_m': 0.004, 'n': 74}
