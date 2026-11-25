# iter14 known issues

1. **City edge wjit 0.054** (iter13 regression) is still in the default. `FBEV_EDGE_CAR_BF=trim` restores 0.037, but z20
   L presence drops at clip start, so it is off.
2. **Night-gate margin:** the KITTI late frames closest to firing have road S 94 and frame-top V 30 (deep-night branch
   needs V < 25; strong-cast branch needs S > 105). The top-V test is image-only. Road S depends on the SAM mask, so any
   new run should check `gate.jsonl` (all `"on": 0` on public clips). Likewise the ego no-op on GPU rests on the
   road-masked estimate staying far outside the guard where iter13 GPU falls back to the prior (CPU masks: sample
   0.79 vs 25, a2d2a 0.14 vs 8, a2d2b 0.34 vs 8 m/s; guard 0.33-3x). In the 8-frame kernel smoke (too few frames for a
   valid BEV phase) exit_approach takes bev_road (22.8 vs 22 prior); full-length runs use bev_phase there.
3. **Fork / merge gores on the private clips:** the night split is not flagged; the day merge gores are not flagged
   (0 false on both). The temporal widen rule only sees the near field. The spatial wedge rule (`FBEV_GORE_SPATIAL=1`)
   finds most split frames, but gives false gores on 4 public clips. This is the iter15 focus.
4. **Private night split, costs of TB / VPLANE:** z30 lane recall 0.81 -> 0.75, lane classification 0.84 -> 0.66; ego
   distance +15.5 % (flat-ground over-range).
5. **Frame 0 incomplete** on lane_split (R lane = split branch), munich (R edge, diagonal-copy veto), a2d2a / a2d2b
   (no lane evidence for 4-16 s).
6. **WildDet3D misses trucks** and gives far-range night false positives (private clips). A truck prompt gave +4 pts
   recall only and was not adopted.
7. As iter13: kerb-pushed measured edges can overlap car footprints (munich 63, a2d2a 44 frames); lane_split lane wjit 0.131.
