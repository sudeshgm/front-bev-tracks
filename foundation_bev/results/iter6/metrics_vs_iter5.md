# Metrics versus merged iter5 (9-clip iter6)

iter5 is the merged 9-clip result at `/workspace/foundation_bev/iter5/` (4 highway clips from the completed iter5 GPU run + 5 recovered clips from iter5b).
iter6 is this full stage 1+2 GPU run. Both scored KITTI and A2D2 at capture_hz (10 fps) and Munich at 30 fps, taken from each clip's polylines JSON.
Command: `smoothness_metrics.py --run <merged iter5> --label iter5 --run <iter6> --label iter6`.
Gore frames are counts of frames with a non-empty `gores` list; sources are `stripes` / `fork` / `widen` (iter5 also had `fork_edge`, which iter6 removed).
Knobs were left at defaults (`FBEV_EDGE_MAX_DEG=15`, `FBEV_GORE_GROW_M=1.2`, `FBEV_GORE_MAX_STEP=0.4`). They were not changed after this comparison.

| clip | accel iter5 → iter6 | n_tracks 5→6 | lane L10/R10 pres % | lane L10/R10 jit2 | lane wjit | edge L10/R10 pres % | edge L10/R10 jit2 | gore frames iter5 → iter6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample | 0.095 → 0.095 | 1→1 | 92.5/92.5 → 95.0/95.0 | 0.007/0.005 → 0.007/0.005 | 0.007 → 0.007 | 95.0/95.0 → 95.0/95.0 | 0.027/0.017 → 0.027/0.017 | 0 → 0 |
| exit_approach | 0.546 → 0.546 | 2→2 | 92.5/92.5 → 95.0/95.0 | 0.007/0.019 → 0.007/0.018 | 0.034 → 0.033 | 95.0/95.0 → 95.0/95.0 | 0.019/0.032 → 0.019/0.032 | 0 → 6 (widen 6; f0021–f0026) |
| late | 0.575 → 0.575 | 5→5 | 87.5/92.5 → 92.5/95.0 | 0.018/0.019 → 0.019/0.016 | 0.022 → 0.020 | 95.0/87.5 → 95.0/77.5 | 0.023/0.040 → 0.026/0.034 | 0 → 0 |
| final | 0.626 → 0.638 | 3→3 | 83.3/86.7 → 93.3/90.0 | 0.033/0.006 → 0.032/0.006 | 0.019 → 0.019 | 90.0/86.7 → 90.0/86.7 | 0.033/0.040 → 0.034/0.040 | 0 → 0 |
| city | 0.566 → 0.566 | 5→5 | 90.0/90.0 → 93.3/93.3 | 0.004/0.011 → 0.004/0.010 | 0.009 → 0.008 | 93.3/93.3 → 93.3/93.3 | 0.064/0.026 → 0.065/0.026 | 12 (fork 8, fork_edge 8; f0018–f0029) → 0 |
| lane_split | 1.074 → **0.997** | 13→13 | 90.0/46.7 → 86.7/66.7 | 0.062/0.021 → 0.047/0.025 | 0.164 → 0.066 | 93.3/93.3 → 40.0/83.3 | 0.069/0.071 → 0.049/0.067 | 19 (fork 18, fork_edge 11; f0009–f0016,f0019–f0029) → 0 |
| munich | 0.911 → 0.912 | 87→87 | 84.9/56.9 → 85.7/65.3 | 0.009/0.004 → 0.009/0.003 | 0.010 → 0.010 | 99.2/92.4 → 87.9/64.0 | 0.034/0.043 → 0.033/0.037 | 368 (fork 58, fork_edge 459; f0009–f0013,f0021–f0059,f0074–f0097,f0109–f0118,f0121–f0125,f0128–f0151,f0241–f0248,f0260–f0272,f0283–f0290,f0306–f0314,f0413–f0419,f0494–f0501,f0507–f0518,f0541–f0546,f0549–f0553,f0577–f0607,f0613–f0707,f0732–f0743,f0749–f0763,f0817–f0838,f0844–f0853) → 6 (fork 6; f0760–f0765) |
| a2d2a | 0.791 → 0.901 | 23→22 | 12.8/22.7 → 11.0/25.0 | 0.001/0.009 → 0.001/0.011 | — → — | 79.7/77.3 → 76.2/71.5 | 0.067/0.060 → 0.064/0.066 | 81 (fork 8, fork_edge 88; f0008–f0028,f0033–f0050,f0064–f0089,f0108–f0123) → 0 |
| a2d2b | 0.890 → **0.827** | 11→11 | 8.7/1.7 → 8.7/1.7 | 0.017/0.050 → 0.014/0.050 | — → — | 97.1/97.1 → 85.5/83.7 | 0.059/0.056 → 0.055/0.052 | 44 (fork 12, fork_edge 59; f0108–f0122,f0129–f0157) → 0 |

## Presence drops > 5 points (flag only; knobs unchanged)

Checked munich edges first, then every other clip's lane and edge L10/R10 presence. A drop is flagged when iter6 presence is more than 5 points below merged iter5. SPEC warned that the new diagonal / crossing edge rule may reduce munich (and other) edge presence; `FBEV_EDGE_MAX_DEG` was left at the default 15.

| clip | kind | side | iter5 % | iter6 % | delta |
| --- | --- | --- | --- | --- | --- |
| munich | edge | L10 | 99.2 | 87.9 | -11.3 |
| munich | edge | R10 | 92.4 | 64.0 | -28.4 |
| late | edge | R10 | 87.5 | 77.5 | -10.0 |
| lane_split | edge | L10 | 93.3 | 40.0 | -53.3 |
| lane_split | edge | R10 | 93.3 | 83.3 | -10.0 |
| a2d2a | edge | R10 | 77.3 | 71.5 | -5.8 |
| a2d2b | edge | L10 | 97.1 | 85.5 | -11.6 |
| a2d2b | edge | R10 | 97.1 | 83.7 | -13.4 |

---

# Smoothness metrics

accel = RMS BEV acceleration of drawn track positions (m/s^2); idsw = probable ID switches /100 frames;
jit = median |frame-to-frame change| of nearest left/right line lateral offset at z (m); pres = % frames present.
jit2 = median |2nd difference| (lane-change aware: steady sideways drift is not jitter); wjit = median |change of half the L-R spacing| (common mode removed). z=12 columns: Munich nearest ground is ~10.4 m.

## iter5

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

## iter6

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
