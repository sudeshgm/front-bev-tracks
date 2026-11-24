# CPU smoke: iter8 code vs iter9 code on the same iter8 GPU stage-1 dets (classical masks, NumPy-1 sim)

Changed metrics only (accel_rms, idsw, n_tracks identical on all 9 clips; all rc 0).

| clip | metric | iter8_code | iter9_code |
|---|---|---|---|
| a2d2a | edges_L10_jit2 | 0.048 | 0.05 |
| a2d2a | edges_R10_jit2 | 0.045 | 0.034 |
| a2d2a | edges_10_wjit | 0.079 | 0.089 |
| a2d2b | edges_L10_jit2 | 0.056 | 0.036 |
| a2d2b | edges_R10_jit2 | 0.057 | 0.036 |
| a2d2b | lanes_R10_jit2 | 0.002 | 0.003 |
| a2d2b | edges_10_wjit | 0.085 | 0.083 |
| kitti_city_cyclist_candidate | edges_L10_jit2 | 0.031 | 0.033 |
| kitti_city_cyclist_candidate | edges_R10_jit2 | 0.042 | 0.023 |
| kitti_city_cyclist_candidate | lanes_L10_jit2 | 0.008 | 0.007 |
| kitti_city_cyclist_candidate | edges_10_wjit | 0.07 | 0.064 |
| kitti_highway_exit_approach | edges_L10_jit2 | 0.025 | 0.016 |
| kitti_highway_exit_approach | edges_R10_jit2 | 0.052 | 0.05 |
| kitti_highway_exit_approach | lanes_L10_jit2 | 0.026 | 0.015 |
| kitti_highway_exit_approach | lanes_R10_jit2 | 0.007 | 0.006 |
| kitti_highway_exit_approach | lanes_10_wjit | 0.018 | 0.009 |
| kitti_highway_exit_approach | edges_10_wjit | 0.052 | 0.043 |
| kitti_highway_final | edges_L10_jit2 | 0.062 | 0.066 |
| kitti_highway_final | edges_R10_jit2 | 0.048 | 0.034 |
| kitti_highway_final | lanes_R10_jit2 | 0.019 | 0.012 |
| kitti_highway_final | lanes_10_wjit | 0.014 | 0.012 |
| kitti_highway_final | edges_10_wjit | 0.066 | 0.05 |
| kitti_highway_late | edges_L10_jit2 | 0.052 | 0.053 |
| kitti_highway_late | edges_R10_jit2 | 0.058 | 0.034 |
| kitti_highway_late | lanes_L10_jit2 | 0.006 | 0.004 |
| kitti_highway_late | lanes_R10_jit2 | 0.003 | 0.007 |
| kitti_highway_late | lanes_10_wjit | 0.022 | 0.009 |
| kitti_highway_late | edges_10_wjit | 0.082 | 0.058 |
| kitti_highway_sample | edges_L10_jit2 | 0.052 | 0.042 |
| kitti_highway_sample | edges_R10_jit2 | 0.018 | 0.012 |
| kitti_highway_sample | edges_10_wjit | 0.059 | 0.046 |
| kitti_lane_split_candidate | edges_L10_jit2 | 0.058 | 0.035 |
| kitti_lane_split_candidate | edges_R10_jit2 | 0.071 | 0.06 |
| kitti_lane_split_candidate | lanes_L10_jit2 | 0.001 | 0.002 |
| kitti_lane_split_candidate | lanes_R10_jit2 | 0.0 | 0.008 |
| kitti_lane_split_candidate | lanes_10_wjit | 0.063 | 0.096 |
| kitti_lane_split_candidate | edges_10_wjit | 0.119 | 0.102 |
| munich | edges_R10_pres | 91.0 | 90.8 |
| munich | lanes_L10_pres | 36.6 | 36.7 |
| munich | lanes_R10_pres | 17.8 | 17.7 |
| munich | edges_L10_jit2 | 0.042 | 0.017 |
| munich | edges_R10_jit2 | 0.046 | 0.029 |
| munich | lanes_L10_jit2 | 0.004 | 0.002 |
| munich | lanes_R10_jit2 | 0.001 | 0.002 |
| munich | edges_10_wjit | 0.071 | 0.05 |

Gores: identical (lane_split f3–7 classical-mask artifact with both codes; no others). Held main-edge records identical: a2d2a 79, a2d2b 28, city 15, exit 7, final 2, late 19, sample 0, lane_split 6, munich 170.
Kerb-rule records (iter9): munich 511, a2d2b 178, a2d2a 86, lane_split 4, KITTI highways/city 0.
