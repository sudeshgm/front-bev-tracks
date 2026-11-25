# iter15 GPU vs iter13 GPU

| clip | metric | gpu13 | gpu15 | YOLO |
|---|---|---|---|---|
| sample | accel | 0.095 | 0.095 |  |
| sample | idsw | 0.000 | 0.000 |  |
| sample | lane pres L/R | 100.0/100.0 | 100.0/100.0 | 100.0/100.0 |
| sample | lane wjit | 0.001 | 0.001 | 0.006 |
| sample | lane jit2 L/R | 0.002/0.002 | 0.002/0.002 | 0.019/0.005 |
| sample | edge pres L/R | 100.0/100.0 | 100.0/100.0 |  |
| sample | edge wjit | 0.019 | 0.019 |  |
| sample | edge jit2 L/R | 0.020/0.009 | 0.020/0.009 |  |
| sample | lane wjit live | 0.001 | 0.001 | |
| sample | edge wjit live | 0.019 | 0.019 | |
| sample | gores | — | — | |
| sample | f0 laneL,R edgeL,R | ✓✓✓✓ | ✓✓✓✓ | |
| exit | accel | 0.546 | 0.546 |  |
| exit | idsw | 0.000 | 0.000 |  |
| exit | lane pres L/R | 100.0/100.0 | 100.0/100.0 | 100.0/100.0 |
| exit | lane wjit | 0.007 | 0.007 | 0.007 |
| exit | lane jit2 L/R | 0.005/0.008 | 0.005/0.010 | 0.013/0.009 |
| exit | edge pres L/R | 100.0/100.0 | 100.0/100.0 |  |
| exit | edge wjit | 0.042 | 0.041 |  |
| exit | edge jit2 L/R | 0.016/0.024 | 0.016/0.026 |  |
| exit | lane wjit live | 0.007 | 0.006 | |
| exit | edge wjit live | 0.048 | 0.045 | |
| exit | gores | 21-26 | 21-26 | |
| exit | f0 laneL,R edgeL,R | ✓✓✓✓ | ✓✓✓✓ | |
| late | accel | 0.534 | 0.534 |  |
| late | idsw | 0.000 | 0.000 |  |
| late | lane pres L/R | 97.5/100.0 | 97.5/100.0 | 100.0/100.0 |
| late | lane wjit | 0.006 | 0.006 | 0.009 |
| late | lane jit2 L/R | 0.011/0.012 | 0.011/0.014 | 0.014/0.014 |
| late | edge pres L/R | 100.0/100.0 | 100.0/100.0 |  |
| late | edge wjit | 0.034 | 0.041 |  |
| late | edge jit2 L/R | 0.020/0.015 | 0.021/0.020 |  |
| late | lane wjit live | 0.005 | 0.006 | |
| late | edge wjit live | 0.035 | 0.041 | |
| late | gores | — | — | |
| late | f0 laneL,R edgeL,R | ✓✓✓✓ | ✓✓✓✓ | |
| final | accel | 0.574 | 0.574 |  |
| final | idsw | 0.000 | 0.000 |  |
| final | lane pres L/R | 100.0/100.0 | 100.0/100.0 | 100.0/100.0 |
| final | lane wjit | 0.006 | 0.008 | 0.008 |
| final | lane jit2 L/R | 0.014/0.004 | 0.027/0.005 | 0.024/0.017 |
| final | edge pres L/R | 100.0/100.0 | 100.0/100.0 |  |
| final | edge wjit | 0.022 | 0.021 |  |
| final | edge jit2 L/R | 0.029/0.023 | 0.031/0.024 |  |
| final | lane wjit live | 0.006 | 0.006 | |
| final | edge wjit live | 0.023 | 0.023 | |
| final | gores | — | — | |
| final | f0 laneL,R edgeL,R | ✓✓✓✓ | ✓✓✓✓ | |
| city | accel | 0.548 | 0.548 |  |
| city | idsw | 0.000 | 0.000 |  |
| city | lane pres L/R | 100.0/100.0 | 100.0/100.0 | 100.0/100.0 |
| city | lane wjit | 0.001 | 0.001 | 0.005 |
| city | lane jit2 L/R | 0.002/0.001 | 0.002/0.001 | 0.001/0.004 |
| city | edge pres L/R | 100.0/100.0 | 100.0/100.0 |  |
| city | edge wjit | 0.054 | 0.054 |  |
| city | edge jit2 L/R | 0.023/0.010 | 0.024/0.010 |  |
| city | lane wjit live | 0.001 | 0.001 | |
| city | edge wjit live | 0.054 | 0.054 | |
| city | gores | — | — | |
| city | f0 laneL,R edgeL,R | ✓✓✓✓ | ✓✓✓✓ | |
| lane_split | accel | 0.951 | 0.951 |  |
| lane_split | idsw | 0.000 | 0.000 |  |
| lane_split | lane pres L/R | 100.0/73.3 | 100.0/73.3 | 76.7/26.7 |
| lane_split | lane wjit | 0.131 | 0.131 | 0.092 |
| lane_split | lane jit2 L/R | 0.011/0.022 | 0.011/0.022 | 0.257/0.721 |
| lane_split | edge pres L/R | 100.0/100.0 | 100.0/100.0 |  |
| lane_split | edge wjit | 0.115 | 0.115 |  |
| lane_split | edge jit2 L/R | 0.042/0.042 | 0.042/0.042 |  |
| lane_split | lane wjit live | 0.131 | 0.131 | |
| lane_split | edge wjit live | 0.127 | 0.127 | |
| lane_split | gores | — | — | |
| lane_split | f0 laneL,R edgeL,R | ✓✗✓✓ | ✓✗✓✓ | |
| munich | accel | 0.655 | 0.655 |  |
| munich | idsw | 0.000 | 0.000 |  |
| munich | lane pres L/R | 85.9/65.4 | 85.9/65.4 | — |
| munich | lane wjit | 0.006 | 0.006 | — |
| munich | lane jit2 L/R | 0.003/0.002 | 0.003/0.002 | — |
| munich | edge pres L/R | 100.0/96.8 | 100.0/96.8 |  |
| munich | edge wjit | 0.037 | 0.037 |  |
| munich | edge jit2 L/R | 0.010/0.020 | 0.010/0.020 |  |
| munich | lane wjit live | 0.006 | 0.006 | |
| munich | edge wjit live | 0.039 | 0.039 | |
| munich | gores | 760-765 | 760-765 | |
| munich | f0 laneL,R edgeL,R | ✓✓✓✗ | ✓✓✓✗ | |
| a2d2a | accel | 0.499 | 0.495 |  |
| a2d2a | idsw | 0.000 | 0.000 |  |
| a2d2a | lane pres L/R | 14.5/27.3 | 15.1/26.7 | — |
| a2d2a | lane wjit | — | — | — |
| a2d2a | lane jit2 L/R | 0.008/0.009 | 0.008/0.007 | — |
| a2d2a | edge pres L/R | 97.7/97.7 | 97.7/98.8 |  |
| a2d2a | edge wjit | 0.067 | 0.062 |  |
| a2d2a | edge jit2 L/R | 0.037/0.032 | 0.036/0.030 |  |
| a2d2a | lane wjit live | — | — | |
| a2d2a | edge wjit live | 0.068 | 0.065 | |
| a2d2a | gores | — | — | |
| a2d2a | f0 laneL,R edgeL,R | ✗✓✓✓ | ✗✓✓✓ | |
| a2d2b | accel | 0.821 | 0.821 |  |
| a2d2b | idsw | 0.000 | 0.000 |  |
| a2d2b | lane pres L/R | 8.100/2.300 | 8.100/2.300 | — |
| a2d2b | lane wjit | — | — | — |
| a2d2b | lane jit2 L/R | 0.012/0.035 | 0.012/0.035 | — |
| a2d2b | edge pres L/R | 99.4/100.0 | 99.4/100.0 |  |
| a2d2b | edge wjit | 0.056 | 0.056 |  |
| a2d2b | edge jit2 L/R | 0.038/0.034 | 0.038/0.032 |  |
| a2d2b | lane wjit live | — | — | |
| a2d2b | edge wjit live | 0.056 | 0.056 | |
| a2d2b | gores | — | — | |
| a2d2b | f0 laneL,R edgeL,R | ✗✗✓✓ | ✗✗✓✓ | |

munich R z10 gpu13 871 -> gpu15 871: gained []  lost []

## Night gate (TB_ADAPT / VPLANE auto): gate-on counts per clip (must be 0)

| clip | gate calls | on | draft-gate passes | road S max | top V min |
|---|---|---|---|---|---|
| a2d2a | 344 | **0** | 40 | 107 | 91 |
| a2d2b | 344 | **0** | 18 | 87 | 99 |
| kitti_city_cyclist_candidate | 60 | **0** | 0 | 62 | 48 |
| kitti_highway_exit_approach | 80 | **0** | 80 | 81 | 160 |
| kitti_highway_final | 60 | **0** | 40 | 111 | 210 |
| kitti_highway_late | 80 | **0** | 12 | 96 | 30 |
| kitti_highway_sample | 80 | **0** | 0 | 32 | 191 |
| kitti_lane_split_candidate | 60 | **0** | 0 | 21 | 42 |
| munich | 1800 | **0** | 0 | 32 | 228 |

TOTAL on = 0 (OK: TB_ADAPT / VPLANE never ran)

## Car audit

```
kitti_highway_sample: synthetic (anchored/backfill/edgehold) inside a car: 0  
kitti_highway_exit_approach: synthetic (anchored/backfill/edgehold) inside a car: 0  
kitti_highway_late: synthetic (anchored/backfill/edgehold) inside a car: 0  
kitti_highway_final: synthetic (anchored/backfill/edgehold) inside a car: 0  
kitti_city_cyclist_candidate: synthetic (anchored/backfill/edgehold) inside a car: 0  live=15[L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12 L13 ...]
kitti_lane_split_candidate: synthetic (anchored/backfill/edgehold) inside a car: 0  held_tracker=2[L18 L19] kerb=2[R15 R16] live=21[R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 ...]
munich: synthetic (anchored/backfill/edgehold) inside a car: 0  held_tracker=18[R4 R5 R6 R7 R12 R162 R474 R475 R650 R651 R656 R657 ...] kerb=62[R14 R15 R16 R17 R28 R29 R32 R33 R40 R41 R42 R43 ...] live=232[R2 R3 R88 R89 R434 R435 R442 R443 R444 R445 R471 R472 ...]
a2d2a: synthetic (anchored/backfill/edgehold) inside a car: 0  held_tracker=19[L4 L5 L6 L12 L13 L20 L27 L28 L59 L92 L93 L94 ...] kerb=44[L15 L53 L60 R68 L72 L73 L74 L75 L76 L77 L78 L79 ...] live=115[L3 L7 L8 L9 L10 L11 L14 L16 L17 L18 L19 L21 ...]
a2d2b: synthetic (anchored/backfill/edgehold) inside a car: 0  held_tracker=12[L10 L11 L38 L40 L70 L71 L82 L83 L89 L91 L107 L108] live=131[L9 L15 L16 L17 L18 L19 L20 L21 L22 L23 L24 L27 ...]
```

## Change scope vs iter13 (tracks / ego pose / lines)

```
kitti_highway_sample           speed  25.00 ->  25.00 | tracks identical | ego pose differs on 39/40 frames | lanes/edges differ on 0/40 frames | header keys changed: []
kitti_highway_exit_approach    speed  23.45 ->  25.07 | tracks identical | ego pose differs on 39/40 frames | lanes/edges differ on 40/40 frames | header keys changed: []
kitti_highway_late             speed  16.65 ->  18.43 | tracks identical | ego pose differs on 39/40 frames | lanes/edges differ on 40/40 frames | header keys changed: []
kitti_highway_final            speed  19.21 ->  20.28 | tracks identical | ego pose differs on 29/30 frames | lanes/edges differ on 30/30 frames | header keys changed: []
kitti_city_cyclist_candidate   speed  11.27 ->  10.81 | tracks identical | ego pose differs on 29/30 frames | lanes/edges differ on 30/30 frames | header keys changed: []
kitti_lane_split_candidate     speed   4.66 ->   4.66 | tracks identical | ego pose differs on 29/30 frames | lanes/edges differ on 0/30 frames | header keys changed: []
munich                         speed  13.13 ->  13.13 | tracks DIFF | ego pose differs on 899/900 frames | lanes/edges differ on 155/900 frames | header keys changed: []
a2d2a                          speed   8.00 ->   8.00 | tracks DIFF | ego pose differs on 170/172 frames | lanes/edges differ on 172/172 frames | header keys changed: ['lane_width_check', 'line_events', 'ground']
a2d2b                          speed   8.00 ->   8.00 | tracks DIFF | ego pose differs on 171/172 frames | lanes/edges differ on 111/172 frames | header keys changed: ['lane_width_check', 'line_events', 'ground']
```

## Ego vs KITTI OXTS (before = iter13 GPU, signfix = iter13 psi negated, after = iter15 GPU)

```
kitti_highway_sample           before   speed  25.00 vs  25.69 ( -2.7%)  yaw corr -0.07 mae  0.51 drift   +1.6  path end  3.27 m (lat  1.84) rms  1.82 / 100.2 m  yaw jerk 4.1
kitti_highway_sample           signfix  speed  25.00 vs  25.69 ( -2.7%)  yaw corr +0.07 mae  0.50 drift   -0.3  path end  2.77 m (lat  0.64) rms  1.63 / 100.2 m  yaw jerk 4.1
kitti_highway_sample           after    speed  25.00 vs  25.69 ( -2.7%)  yaw corr +0.68 mae  0.11 drift   +0.2  path end  2.69 m (lat  0.03) rms  1.58 / 100.2 m  yaw jerk 0.7
kitti_highway_exit_approach    before   speed  23.45 vs  24.44 ( -4.0%)  yaw corr -0.36 mae  1.40 drift   -1.5  path end  3.92 m (lat  0.11) rms  2.75 /  95.4 m  yaw jerk 5.3
kitti_highway_exit_approach    signfix  speed  23.45 vs  24.44 ( -4.0%)  yaw corr +0.36 mae  1.56 drift   +3.6  path end  4.32 m (lat  1.83) rms  2.86 /  95.4 m  yaw jerk 5.3
kitti_highway_exit_approach    after    speed  25.07 vs  24.44 ( +2.6%)  yaw corr +0.24 mae  0.13 drift   -0.1  path end  2.39 m (lat  0.07) rms  1.01 /  95.4 m  yaw jerk 0.8
kitti_highway_late             before   speed  16.65 vs  18.35 ( -9.2%)  yaw corr -0.78 mae  3.48 drift   -1.9  path end  6.12 m (lat  1.91) rms  2.96 /  70.7 m  yaw jerk 13.5
kitti_highway_late             signfix  speed  16.65 vs  18.35 ( -9.2%)  yaw corr +0.78 mae  1.42 drift   +4.7  path end  6.00 m (lat  1.47) rms  2.90 /  70.7 m  yaw jerk 13.5
kitti_highway_late             after    speed  18.43 vs  18.35 ( +0.4%)  yaw corr +0.66 mae  0.86 drift   +0.8  path end  1.35 m (lat  0.81) rms  1.53 /  70.7 m  yaw jerk 7.0
kitti_highway_final            before   speed  19.21 vs  19.51 ( -1.5%)  yaw corr -0.35 mae  0.59 drift   -0.8  path end  1.35 m (lat  0.93) rms  0.60 /  56.7 m  yaw jerk 2.5
kitti_highway_final            signfix  speed  19.21 vs  19.51 ( -1.5%)  yaw corr +0.35 mae  0.59 drift   -1.8  path end  1.38 m (lat  0.98) rms  0.61 /  56.7 m  yaw jerk 2.5
kitti_highway_final            after    speed  20.28 vs  19.51 ( +4.0%)  yaw corr +0.86 mae  0.21 drift   +0.5  path end  2.12 m (lat  0.17) rms  1.48 /  56.7 m  yaw jerk 1.3
kitti_city_cyclist_candidate   before   speed  11.27 vs  10.20 (+10.5%)  yaw corr -0.72 mae  0.88 drift   +0.1  path end  3.36 m (lat  0.42) rms  1.52 /  29.4 m  yaw jerk 2.3
kitti_city_cyclist_candidate   signfix  speed  11.27 vs  10.20 (+10.5%)  yaw corr +0.72 mae  1.28 drift   -3.7  path end  3.48 m (lat  1.00) rms  1.57 /  29.4 m  yaw jerk 2.3
kitti_city_cyclist_candidate   after    speed  10.81 vs  10.20 ( +6.0%)  yaw corr +0.77 mae  0.41 drift   -0.8  path end  2.00 m (lat  0.18) rms  0.78 /  29.4 m  yaw jerk 1.5
kitti_lane_split_candidate     before   speed   4.66 vs   5.68 (-18.0%)  yaw corr +0.50 mae 24.42 drift  -71.3  path end  9.84 m (lat  9.48) rms  4.67 /  15.9 m  yaw jerk 11.1
kitti_lane_split_candidate     signfix  speed   4.66 vs   5.68 (-18.0%)  yaw corr -0.50 mae  2.28 drift   +7.2  path end  2.71 m (lat  0.66) rms  1.46 /  15.9 m  yaw jerk 11.1
kitti_lane_split_candidate     after    speed   4.66 vs   5.68 (-18.0%)  yaw corr +0.75 mae  1.13 drift   +3.2  path end  2.69 m (lat  0.57) rms  1.47 /  15.9 m  yaw jerk 9.7
```

## Ego speed sources + lane pitch

- a2d2a: prior 8.00 m/s; yaw bev; lane pitch {'applied': False, 'n_pairs': 0, 'ratio': None, 'theta_deg': None}
- a2d2b: prior 8.00 m/s; yaw bev; lane pitch {'applied': False, 'n_pairs': 0, 'ratio': None, 'theta_deg': None}
- kitti_city_cyclist_candidate: bev_phase 10.81 m/s; yaw bev; lane pitch {'applied': True, 'n_pairs': 39, 'ratio': 1.0238, 'theta_deg': 0.126}
- kitti_highway_exit_approach: bev_phase 25.07 m/s; yaw bev; lane pitch {'applied': True, 'n_pairs': 40, 'ratio': 0.958, 'theta_deg': -0.216}
- kitti_highway_final: bev_phase 20.28 m/s; yaw bev; lane pitch {'applied': True, 'n_pairs': 8, 'ratio': 0.9413, 'theta_deg': -0.298}
- kitti_highway_late: bev_phase 18.43 m/s; yaw bev; lane pitch {'applied': True, 'n_pairs': 6, 'ratio': 0.9259, 'theta_deg': -0.374}
- kitti_highway_sample: prior 25.00 m/s; yaw bev; lane pitch {'applied': True, 'n_pairs': 22, 'ratio': 0.9616, 'theta_deg': -0.198}
- kitti_lane_split_candidate: bev_phase 4.66 m/s; yaw bev; lane pitch {'applied': False, 'n_pairs': 0, 'ratio': None, 'theta_deg': None}
- munich: scene 13.13 m/s; yaw bev; lane pitch {'applied': True, 'n_pairs': 10, 'ratio': 1.1784, 'theta_deg': 0.945}
