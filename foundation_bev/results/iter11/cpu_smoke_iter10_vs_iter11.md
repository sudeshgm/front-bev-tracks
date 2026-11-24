# CPU smoke (classical masks, iter10 GPU dets, NumPy-1 sim): iter10 code → iter11 code

| clip | accel_rms_mps2 | id_switch_proxy_per100f | lanes_L10_pres | lanes_R10_pres | lanes_10_wjit | lanes_L10_jit2 | lanes_R10_jit2 | edges_L10_pres | edges_R10_pres | edges_10_wjit | edges_R10_jit2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a2d2a | 0.819 → 0.819 | 0.0 → 0.0 | 0.0 → 0.0 | 0.0 → 0.0 | None → None | None → None | None → None | 80.2 → 82.6 | 94.2 → 95.3 | 0.088 → 0.092 | 0.039 → 0.041 |
| a2d2b | 0.622 → 0.622 | 0.0 → 0.0 | 0.0 → 0.0 | 21.5 → 21.5 | None → None | None → None | 0.003 → 0.003 | 95.3 → 95.3 | 97.1 → 97.1 | 0.085 → 0.085 | 0.037 → 0.037 |
| kitti_city_cyclist_candidate | 0.548 → 0.548 | 0.0 → 0.0 | 80.0 → 80.0 | 0.0 → 0.0 | None → None | 0.007 → 0.007 | None → None | 73.3 → 93.3 | 76.7 → 76.7 | 0.056 → 0.056 | 0.021 → 0.021 |
| kitti_highway_exit_approach | 0.492 → 0.492 | 0.0 → 0.0 | 22.5 → 22.5 | 100.0 → 100.0 | 0.011 → 0.011 | 0.018 → 0.018 | 0.006 → 0.006 | 90.0 → 100.0 | 95.0 → 100.0 | 0.045 → 0.046 | 0.053 → 0.055 |
| kitti_highway_final | 0.517 → 0.517 | 0.0 → 0.0 | 76.7 → 90.0 | 80.0 → 100.0 | 0.011 → 0.007 | 0.001 → 0.001 | 0.011 → 0.005 | 90.0 → 90.0 | 80.0 → 100.0 | 0.054 → 0.052 | 0.019 → 0.015 |
| kitti_highway_late | 0.645 → 0.645 | 0.0 → 0.0 | 65.0 → 65.0 | 45.0 → 45.0 | 0.012 → 0.012 | 0.004 → 0.004 | 0.007 → 0.007 | 95.0 → 100.0 | 95.0 → 100.0 | 0.062 → 0.058 | 0.026 → 0.023 |
| kitti_highway_sample | 0.196 → 0.196 | 0.0 → 0.0 | 0.0 → 0.0 | 100.0 → 100.0 | None → None | None → None | 0.005 → 0.005 | 90.0 → 100.0 | 90.0 → 100.0 | 0.039 → 0.033 | 0.011 → 0.01 |
| kitti_lane_split_candidate | 0.817 → 0.817 | 0.0 → 0.0 | 56.7 → 56.7 | 60.0 → 60.0 | 0.053 → 0.053 | 0.002 → 0.002 | 0.004 → 0.004 | 86.7 → 100.0 | 86.7 → 100.0 | 0.138 → 0.13 | 0.066 → 0.066 |
| munich | 0.649 → 0.649 | 0.0 → 0.0 | 37.9 → 37.9 | 18.6 → 18.6 | 0.006 → 0.006 | 0.002 → 0.002 | 0.002 → 0.002 | 98.1 → 98.3 | 69.3 → 69.9 | 0.059 → 0.059 | 0.025 → 0.025 |

## gores / held / kerb_occ / backfilled edge records

| clip | gores iter10 code | gores iter11 code | held | kerb_occ | edge backfill |
|---|---|---|---|---|---|
| a2d2a | - | - | 52 → 52 | 102 → 102 | 0 → 6 |
| a2d2b | - | - | 21 → 21 | 183 → 183 | 0 → 6 |
| kitti_city_cyclist_candidate | - | - | 14 → 14 | 0 → 0 | 0 → 6 |
| kitti_highway_exit_approach | - | - | 7 → 7 | 0 → 0 | 0 → 6 |
| kitti_highway_final | - | - | 2 → 2 | 0 → 0 | 0 → 9 |
| kitti_highway_late | - | - | 26 → 26 | 0 → 0 | 0 → 4 |
| kitti_highway_sample | - | - | 0 → 0 | 0 → 0 | 0 → 8 |
| kitti_lane_split_candidate | stripes: 3-29 (8) | stripes: 3-29 (8) | 6 → 6 | 4 → 4 | 0 → 8 |
| munich | stripes: 32-45 (16); fork: 423-881 (19) | stripes: 32-45 (16); fork: 423-881 (19) | 223 → 226 | 546 → 546 | 0 → 8 |
