# iter2 tracks replayed through iter3 track filters

ego motion: scene speed, straight

scene ego speed (speed, n tracks, accepted, prior): {"a2d2a": {"median": 8.0, "p10": 5.2, "p90": 10.1, "valid_frac": 0.66, "prior": 8.0}, "a2d2b": {"median": 6.0, "p10": 4.7, "p90": 6.8, "valid_frac": 0.71, "prior": 8.0}, "kitti_city_cyclist_candidate": {"median": 10.0, "p10": 7.9, "p90": 10.0, "valid_frac": 0.07, "prior": 10.0}, "kitti_highway_exit_approach": {"median": 22.0, "p10": 22.0, "p90": 22.0, "valid_frac": 0.0, "prior": 22.0}, "kitti_highway_final": {"median": 25.0, "p10": 25.0, "p90": 25.0, "valid_frac": 0.0, "prior": 25.0}, "kitti_highway_late": {"median": 25.0, "p10": 25.0, "p90": 25.0, "valid_frac": 0.0, "prior": 25.0}, "kitti_highway_sample": {"median": 25.0, "p10": 25.0, "p90": 25.0, "valid_frac": 0.0, "prior": 25.0}, "kitti_lane_split_candidate": {"median": 18.0, "p10": 12.7, "p90": 18.0, "valid_frac": 0.13, "prior": 18.0}, "munich": {"median": 12.7, "p10": 7.5, "p90": 14.3, "valid_frac": 0.91, "prior": 12.0}}

| clip | tracks | drawn moving | drawn static | drop short | drop score | drop bicycle no rider | drop off-road VRU | accel RMS iter2 (all) | accel RMS iter3 (moving) |
|---|---|---|---|---|---|---|---|---|---|
| a2d2a | 49 | 27 | 9 | 0 | 10 | 3 | 0 | 1.26 | 1.46 |
| a2d2b | 76 | 46 | 10 | 0 | 14 | 6 | 0 | 1.12 | 1.26 |
| kitti_city_cyclist_candidate | 10 | 9 | 0 | 0 | 0 | 1 | 0 | 0.47 | 0.50 |
| kitti_highway_exit_approach | 6 | 2 | 0 | 0 | 1 | 3 | 0 | 0.63 | 0.69 |
| kitti_highway_final | 9 | 5 | 0 | 0 | 3 | 0 | 1 | 0.84 | 0.95 |
| kitti_highway_late | 13 | 8 | 0 | 0 | 2 | 1 | 2 | 1.70 | 1.96 |
| kitti_highway_sample | 5 | 2 | 0 | 0 | 0 | 3 | 0 | 0.50 | 0.69 |
| kitti_lane_split_candidate | 15 | 14 | 0 | 0 | 0 | 1 | 0 | 1.13 | 1.05 |
| munich | 171 | 60 | 48 | 33 | 14 | 16 | 0 | 1.08 | 1.30 |
