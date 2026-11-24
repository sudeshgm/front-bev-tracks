# iter2 tracks replayed through iter3 track filters

ego motion: prior speed, straight

scene ego speed (speed, n tracks, accepted, prior): {}

| clip | tracks | drawn moving | drawn static | drop short | drop score | drop bicycle no rider | drop off-road VRU | accel RMS iter2 (all) | accel RMS iter3 (moving) |
|---|---|---|---|---|---|---|---|---|---|
| a2d2a | 49 | 31 | 5 | 0 | 10 | 3 | 0 | 1.26 | 1.41 |
| a2d2b | 76 | 51 | 5 | 0 | 14 | 6 | 0 | 1.12 | 1.22 |
| kitti_city_cyclist_candidate | 10 | 9 | 0 | 0 | 0 | 1 | 0 | 0.47 | 0.50 |
| kitti_highway_exit_approach | 6 | 2 | 0 | 0 | 1 | 3 | 0 | 0.63 | 0.69 |
| kitti_highway_final | 9 | 5 | 0 | 0 | 3 | 0 | 1 | 0.84 | 0.95 |
| kitti_highway_late | 13 | 8 | 0 | 0 | 2 | 1 | 2 | 1.70 | 1.96 |
| kitti_highway_sample | 5 | 2 | 0 | 0 | 0 | 3 | 0 | 0.50 | 0.69 |
| kitti_lane_split_candidate | 15 | 14 | 0 | 0 | 0 | 1 | 0 | 1.13 | 1.05 |
| munich | 171 | 91 | 17 | 33 | 14 | 16 | 0 | 1.08 | 1.22 |
