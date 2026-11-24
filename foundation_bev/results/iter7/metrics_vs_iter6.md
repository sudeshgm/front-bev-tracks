# Metrics versus iter6 (9-clip iter7)

iter6 is the completed GPU run at `/workspace/foundation_bev/iter6/`.
iter7 is this full stage 1+2 GPU run (`/workspace/foundation_bev/iter7/`, pulled from volume `gpubot-bev-iter7`). Both scored KITTI and A2D2 at capture_hz (10 fps) and Munich at 30 fps, taken from each clip's polylines JSON and `metrics.json`.
Gore frames are counts of frames with a non-empty `gores` list; sources are `stripes` / `fork` / `widen`.
Held-edge counts are the number of frames with at least one edge that has `held: true`.
Knobs were left at code defaults (`FBEV_EDGE_MAX_DEG=25`, `FBEV_LANE_EDGE_DUP_M=0.8`, `FBEV_GORE_GROW_M=1.2`, `FBEV_GORE_MAX_STEP=0.4`). `FBEV_RAY_R` was unset (off). They were not changed after this comparison.

| clip | accel iter6 → iter7 | n_tracks 6→7 | idsw 6→7 | lane L10/R10 pres % | lane L10/R10 jit2 | lane wjit | edge L10/R10 pres % | edge L10/R10 jit2 | gore frames iter6 → iter7 | held-edge frames |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample | 0.095 → 0.095 | 1→1 | 0.0→0.0 | 95.0/95.0 → 95.0/95.0 | 0.007/0.005 → 0.007/0.005 | 0.007 → 0.007 | 95.0/95.0 → 95.0/95.0 | 0.027/0.017 → 0.026/0.018 | 0 → 0 | 0 |
| exit_approach | 0.546 → 0.546 | 2→2 | 0.0→0.0 | 95.0/95.0 → 95.0/95.0 | 0.007/0.018 → 0.007/0.018 | 0.033 → 0.033 | 95.0/95.0 → 95.0/95.0 | 0.019/0.032 → 0.019/0.033 | 6 (widen 6; f0021–f0026) → 6 (widen 6; f0021–f0026) | 0 |
| late | 0.575 → 0.526 | 5→5 | 0.0→0.0 | 92.5/95.0 → 92.5/95.0 | 0.019/0.016 → 0.012/0.019 | 0.020 → 0.020 | 95.0/77.5 → 95.0/87.5 | 0.026/0.034 → 0.024/0.040 | 0 → 0 | 9 |
| final | 0.638 → 0.614 | 3→3 | 0.0→0.0 | 93.3/90.0 → 93.3/90.0 | 0.032/0.006 → 0.039/0.006 | 0.019 → 0.024 | 90.0/86.7 → 90.0/86.7 | 0.034/0.040 → 0.033/0.040 | 0 → 0 | 0 |
| city | 0.566 → 0.566 | 5→5 | 0.0→0.0 | 93.3/93.3 → 93.3/93.3 | 0.004/0.010 → 0.004/0.010 | 0.008 → 0.008 | 93.3/93.3 → 93.3/93.3 | 0.065/0.026 → 0.065/0.026 | 0 → 0 | 2 |
| lane_split | 0.997 → **1.048** | 13→13 | 0.0→0.0 | 86.7/66.7 → 93.3/73.3 | 0.047/0.025 → 0.063/0.025 | 0.066 → 0.136 | 40.0/83.3 → 93.3/83.3 | 0.049/0.067 → 0.069/0.072 | 0 → 0 | 4 |
| munich | 0.912 → **0.676** | 87→87 | 0.0→0.0 | 85.7/65.3 → 85.7/65.3 | 0.009/0.003 → 0.009/0.003 | 0.010 → 0.010 | 87.9/64.0 → 93.9/72.7 | 0.033/0.037 → 0.034/0.039 | 6 (fork 6; f0760–f0765) → 6 (fork 6; f0760–f0765) | 169 |
| a2d2a | 0.901 → **0.775** | 22→23 | 0.0→0.0 | 11.0/25.0 → 14.5/23.3 | 0.001/0.011 → 0.001/0.010 | — → — | 76.2/71.5 → 96.5/94.2 | 0.064/0.066 → 0.047/0.047 | 0 → 0 | 33 |
| a2d2b | 0.827 → **0.879** | 11→11 | 0.0→0.0 | 8.7/1.7 → 8.7/1.7 | 0.014/0.050 → 0.017/0.050 | — → — | 85.5/83.7 → 97.1/95.9 | 0.055/0.052 → 0.058/0.057 | 0 → 0 | 37 |

## Presence drops > 5 points vs iter6 (flag only; knobs unchanged)

A drop is flagged when iter7 lane or edge L10/R10 presence is more than 5 points below iter6, per clip and side.

**None.** No clip/side dropped more than 5 points versus iter6.

## Edge presence below merged iter5 (STATE.md flag; knobs unchanged)

Flag any clip whose edge L10/R10 presence is below the merged iter5 result in `/workspace/foundation_bev/iter5/` (iter5 highway clips + iter5b recovered clips). Priority checks requested first: lane_split L, munich R, a2d2a accel.

- lane_split L10 pres: iter5 90.0 → iter6 86.7 → iter7 93.3 (recovered vs iter6; at/above iter5).
- munich R10 edge pres: iter5 92.4 → iter6 64.0 → iter7 72.7 (still below iter5).
- a2d2a accel_rms: iter5 0.791 → iter6 0.901 → iter7 0.775.

| clip | kind | side | iter5 % | iter7 % | delta |
| --- | --- | --- | --- | --- | --- |
| munich | edge | R10 | 92.4 | 72.7 | -19.7 |
| munich | edge | L10 | 99.2 | 93.9 | -5.3 |
| lane_split | edge | R10 | 93.3 | 83.3 | -10.0 |
| a2d2b | edge | R10 | 97.1 | 95.9 | -1.2 |

## Gore-frame check

SPEC: gores must still fire only on exit_approach f21–26 and munich f760–765.
**Pass.** Gore frames fired only on exit_approach f0021–f0026 (widen) and munich f0760–f0765 (fork). No other clip had a gore.

## Held-edge frames per clip

| clip | frames with a `held: true` edge | frame list (ranges) |
| --- | --- | --- |
| sample | 0 | — |
| exit_approach | 0 | — |
| late | 9 | f0014,f0021–f0022,f0033–f0036,f0038–f0039 |
| final | 0 | — |
| city | 2 | f0028–f0029 |
| lane_split | 4 | f0005–f0006,f0018–f0019 |
| munich | 169 | f0004–f0007,f0009–f0013,f0044,f0054–f0055,f0062–f0067,f0120–f0121,f0136–f0141,f0144–f0147,f0152–f0153,f0156–f0157,f0160–f0166,f0170–f0178,f0202–f0203,f0246–f0247,f0249,f0253–f0254,f0259–f0260,f0264–f0265,f0290–f0293,f0298–f0301,f0336–f0337,f0360–f0361,f0414–f0415,f0422–f0423,f0432–f0433,f0436–f0437,f0456–f0457,f0474–f0475,f0512–f0513,f0515–f0516,f0518–f0524,f0534–f0535,f0544–f0545,f0560–f0567,f0584–f0587,f0598–f0600,f0602–f0603,f0610–f0611,f0616–f0617,f0622–f0623,f0641–f0643,f0648–f0653,f0656–f0657,f0660–f0665,f0668–f0671,f0688–f0689,f0692–f0693,f0700–f0704,f0706–f0707,f0858–f0859,f0866–f0867,f0870–f0873,f0876–f0877,f0896–f0899 |
| a2d2a | 33 | f0004–f0006,f0008–f0009,f0011–f0013,f0015–f0017,f0020,f0022,f0024,f0027–f0028,f0059,f0062,f0067–f0068,f0092–f0095,f0108–f0111,f0130,f0138–f0139,f0144,f0146 |
| a2d2b | 37 | f0007–f0008,f0010–f0015,f0038–f0040,f0068–f0071,f0082–f0083,f0088–f0089,f0091,f0101–f0104,f0107–f0108,f0126–f0129,f0143–f0145,f0148–f0151 |

---

# Smoothness metrics

accel = RMS BEV acceleration of drawn track positions (m/s^2); idsw = probable ID switches /100 frames;
jit = median |frame-to-frame change| of nearest left/right line lateral offset at z (m); pres = % frames present.
jit2 = median |2nd difference| (lane-change aware: steady sideways drift is not jitter); wjit = median |change of half the L-R spacing| (common mode removed). z=12 columns: Munich nearest ground is ~10.4 m.

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

## iter7

| clip | fps | frames | n_tracks | accel_rms_mps2 | accel_rms_raw_mps2 | id_switch_proxy_per100f | lanes_L10_jit | lanes_R10_jit | lanes_L20_jit | lanes_R20_jit | lanes_L10_pres | lanes_R10_pres | edges_L10_jit | edges_R10_jit | edges_L20_jit | edges_R20_jit | edges_L10_pres | edges_R10_pres | lanes_L10_jit2 | lanes_R10_jit2 | lanes_10_wjit | lanes_L12_pres | lanes_R12_pres | lanes_L12_jit2 | lanes_R12_jit2 | edges_L10_jit2 | edges_R10_jit2 | edges_10_wjit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a2d2a | 10.0 | 172 | 23 | 0.775 | 160.752 | 0.0 | 0.03 | 0.033 | 0.012 | 0.031 | 14.5 | 23.3 | 0.065 | 0.084 | 0.068 | 0.084 | 96.5 | 94.2 | 0.001 | 0.01 |  | 14.5 | 66.9 | 0.0 | 0.007 | 0.047 | 0.047 | 0.064 |
| a2d2b | 10.0 | 172 | 11 | 0.879 | 93.51 | 0.0 | 0.044 | 0.025 | 0.03 | 0.058 | 8.7 | 1.7 | 0.085 | 0.091 | 0.09 | 0.121 | 97.1 | 95.9 | 0.017 | 0.05 |  | 9.3 | 11.6 | 0.013 | 0.019 | 0.058 | 0.057 | 0.085 |
| kitti_city_cyclist_candidate | 10.0 | 30 | 5 | 0.566 | 153.61 | 0.0 | 0.01 | 0.015 | 0.021 | 0.05 | 93.3 | 93.3 | 0.117 | 0.022 | 0.056 | 0.05 | 93.3 | 93.3 | 0.004 | 0.01 | 0.008 | 93.3 | 93.3 | 0.006 | 0.016 | 0.065 | 0.026 | 0.07 |
| kitti_highway_exit_approach | 10.0 | 40 | 2 | 0.546 | 275.954 | 0.0 | 0.011 | 0.034 | 0.013 | 0.053 | 95.0 | 95.0 | 0.039 | 0.073 | 0.024 | 0.072 | 95.0 | 95.0 | 0.007 | 0.018 | 0.033 | 95.0 | 95.0 | 0.007 | 0.018 | 0.019 | 0.033 | 0.055 |
| kitti_highway_final | 10.0 | 30 | 3 | 0.614 | 370.17 | 0.0 | 0.054 | 0.017 | 0.076 | 0.016 | 93.3 | 90.0 | 0.064 | 0.058 | 0.07 | 0.047 | 90.0 | 86.7 | 0.039 | 0.006 | 0.024 | 66.7 | 90.0 | 0.056 | 0.006 | 0.033 | 0.04 | 0.033 |
| kitti_highway_late | 10.0 | 40 | 5 | 0.526 | 365.63 | 0.0 | 0.088 | 0.071 | 0.059 | 0.093 | 92.5 | 95.0 | 0.074 | 0.111 | 0.093 | 0.111 | 95.0 | 87.5 | 0.012 | 0.019 | 0.02 | 90.0 | 95.0 | 0.016 | 0.028 | 0.024 | 0.04 | 0.052 |
| kitti_highway_sample | 10.0 | 40 | 1 | 0.095 | 509.386 | 0.0 | 0.011 | 0.005 | 0.014 | 0.007 | 95.0 | 95.0 | 0.053 | 0.024 | 0.049 | 0.026 | 95.0 | 95.0 | 0.007 | 0.005 | 0.007 | 95.0 | 95.0 | 0.008 | 0.004 | 0.026 | 0.018 | 0.027 |
| kitti_lane_split_candidate | 10.0 | 30 | 13 | 1.048 | 275.84 | 0.0 | 0.052 | 0.144 | 0.056 | 0.05 | 93.3 | 73.3 | 0.213 | 0.071 | 0.333 | 0.274 | 93.3 | 83.3 | 0.063 | 0.025 | 0.136 | 93.3 | 50.0 | 0.065 | 0.13 | 0.069 | 0.072 | 0.14 |
| munich | 30.0 | 900 | 87 | 0.676 | 1891.94 | 0.0 | 0.018 | 0.007 | 0.017 | 0.009 | 85.7 | 65.3 | 0.059 | 0.081 | 0.057 | 0.07 | 93.9 | 72.7 | 0.009 | 0.003 | 0.01 | 92.3 | 84.3 | 0.008 | 0.003 | 0.034 | 0.039 | 0.049 |

coefficient jitter (motion-compensated, median per frame): lanes / edges

- a2d2a: lanes {'dc0_m': 0.033, 'dc1x10_m': 0.001, 'dc2x100_m': 0.0, 'n': 183}; edges {'dc0_m': 0.12, 'dc1x10_m': 0.101, 'dc2x100_m': 0.021, 'n': 324}
- a2d2b: lanes {'dc0_m': 0.056, 'dc1x10_m': 0.008, 'dc2x100_m': 0.001, 'n': 363}; edges {'dc0_m': 0.166, 'dc1x10_m': 0.139, 'dc2x100_m': 0.027, 'n': 342}
- kitti_city_cyclist_candidate: lanes {'dc0_m': 0.026, 'dc1x10_m': 0.03, 'dc2x100_m': 0.005, 'n': 114}; edges {'dc0_m': 0.119, 'dc1x10_m': 0.11, 'dc2x100_m': 0.024, 'n': 54}
- kitti_highway_exit_approach: lanes {'dc0_m': 0.031, 'dc1x10_m': 0.022, 'dc2x100_m': 0.004, 'n': 74}; edges {'dc0_m': 0.091, 'dc1x10_m': 0.034, 'dc2x100_m': 0.006, 'n': 74}
- kitti_highway_final: lanes {'dc0_m': 0.037, 'dc1x10_m': 0.026, 'dc2x100_m': 0.0, 'n': 50}; edges {'dc0_m': 0.065, 'dc1x10_m': 0.032, 'dc2x100_m': 0.004, 'n': 51}
- kitti_highway_late: lanes {'dc0_m': 0.057, 'dc1x10_m': 0.032, 'dc2x100_m': 0.008, 'n': 70}; edges {'dc0_m': 0.138, 'dc1x10_m': 0.058, 'dc2x100_m': 0.015, 'n': 70}
- kitti_highway_sample: lanes {'dc0_m': 0.009, 'dc1x10_m': 0.008, 'dc2x100_m': 0.002, 'n': 74}; edges {'dc0_m': 0.034, 'dc1x10_m': 0.017, 'dc2x100_m': 0.004, 'n': 74}
- kitti_lane_split_candidate: lanes {'dc0_m': 0.086, 'dc1x10_m': 0.0, 'dc2x100_m': 0.0, 'n': 103}; edges {'dc0_m': 0.315, 'dc1x10_m': 0.268, 'dc2x100_m': 0.03, 'n': 52}
- munich: lanes {'dc0_m': 0.031, 'dc1x10_m': 0.018, 'dc2x100_m': 0.002, 'n': 2871}; edges {'dc0_m': 0.103, 'dc1x10_m': 0.066, 'dc2x100_m': 0.012, 'n': 1717}

