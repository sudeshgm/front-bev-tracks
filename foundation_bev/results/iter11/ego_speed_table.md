# EGO SPEED COMPARISON (m/s)

From the iter11 GPU run (`run_meta.ego_speed_table`). Units are m/s.
Column order: prior | flow median (valid) | scene closing-rate median (valid) | **BEV phase (valid, repeats)** | oxts | selected source -> speed.

| clip | prior | flow median (valid) | scene closing-rate median (valid) | BEV phase (valid, repeats) | oxts | selected source -> speed |
| --- | --- | --- | --- | --- | --- | --- |
| kitti_highway_sample | 25.0 | 0.4 (0.70) | 25.0 (0.00) | 25.8 (0.33, 0) | None | prior -> 25.0 |
| kitti_highway_exit_approach | 22.0 | 1.5 (0.95) | 22.0 (0.00) | 23.5 (0.64, 0) | None | bev_phase -> 23.5 |
| kitti_highway_late | 25.0 | 5.5 (0.78) | 25.0 (0.00) | 16.7 (0.92, 0) | None | bev_phase -> 16.7 |
| kitti_highway_final | 25.0 | 1.6 (0.27) | 25.0 (0.00) | 19.2 (0.86, 0) | None | bev_phase -> 19.2 |
| kitti_city_cyclist_candidate | 10.0 | 10.0 (0.97) | 10.0 (0.00) | 11.3 (1.00, 0) | None | bev_phase -> 11.3 |
| kitti_lane_split_candidate | 18.0 | 4.6 (0.97) | 18.0 (0.00) | 4.7 (0.83, 0) | None | bev_phase -> 4.7 |
| munich | 12.0 | 0.5 (0.54) | 13.1 (0.76) | 20.0 (0.94, 450) | None | scene -> 13.1 |
| a2d2a | 8.0 | 0.1 (0.51) | 4.5 (0.13) | 0.1 (0.65, 12) | None | prior -> 8.0 |
| a2d2b | 8.0 | 0.7 (0.67) | 8.0 (0.00) | 0.3 (0.77, 0) | None | prior -> 8.0 |
