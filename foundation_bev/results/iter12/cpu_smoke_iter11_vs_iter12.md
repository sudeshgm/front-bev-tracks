# iter12 CPU smoke vs iter11 code (same inputs)

Inputs: iter11 GPU WD3D dets (`iter11/dets/wd3d_*.npz`) → stage 2 on CPU (gates, tracker, BevPhaseEgo, lines_mt) with
**classical bev_ipm masks as the SAM stand-in** (`--smoke --classical-masks`), all 9 clips, full length.
`run_cpu_smoke.sh` (iter12 code) vs `run_iter11_cpu.sh` (iter11_spec code). 18/18 rc=0. Full table: `cmp_cpu.md`.
The stand-in masks are much noisier than SAM, so these numbers check robustness and code paths, not quality.

| clip | accel | idsw | edge pres L/R iter11 → iter12 | edge wjit 11 → 12 | edge jit2 L/R 11 → 12 | f0 edges L,R 11 → 12 | gores |
|---|---|---|---|---|---|---|---|
| sample | 0.196 = | 0 | 100/100 = | 0.035 → 0.034 | 0.034/0.008 → 0.035/0.008 | ✓✓ = | none = |
| exit | 0.492 = | 0 | 100/100 → 100/**95.0** | 0.044 → 0.043 | 0.016/0.051 → 0.017/0.044 | ✓✓ → ✓✗ | none = |
| late | 0.645 = | 0 | 100/100 = | 0.053 → **0.038** | 0.031/0.021 → 0.033/0.026 | ✓✓ = | none = |
| final | 0.517 = | 0 | 100/100 = | 0.063 = | 0.065/0.014 → 0.066/0.015 | ✓✓ = | none = |
| city | 0.548 = | 0 | 93.3/76.7 → **100/100** | 0.056 = | 0.025/0.021 → 0.028/0.019 | ✓✗ → ✓✓ | none = |
| lane_split | 0.817 = | 0 | 100/100 = | 0.129 = | 0.028/0.066 → 0.025/0.063 | ✓✓ = | identical (stripe) |
| munich | 0.649 = | 0 | 97.6/69.6 → 97.6/70.6 | 0.057 = | 0.011/0.024 → 0.012/0.026 | ✓✗ = | identical |
| a2d2a | 0.819 = | 0 | 82.6/95.3 → **80.2/94.8** | 0.092 → 0.090 | 0.087/0.041 → 0.088/0.041 | ✓✓ → ✗✗ | none = |
| a2d2b | 0.622 = | 0 | 94.2/97.7 → **100/99.4** | 0.085 → 0.079 | 0.036/0.035 → 0.033/0.035 | ✗✗ → ✓✓ | none = |

Lane presence, lane wjit, lane jit2, accel, idsw and gores are identical on every clip. Munich R z10 626 → 635 (+9:
f145, f158, f583-589; lost none). Largest edge jit2 rise: late R +0.005.

## The three edge losses are rejected garbage copies (checked)
The same post-tracker replay on this smoke run with each iter12 feature toggled (`evidence/feature_toggles.md`;
columns: smoke12, iter12 replay, then one feature reverted at a time, then all reverted) shows that all three losses
come from `EDGE_BACKFILL_COVER=1` (diagonal-copy gate + source-must-cover-z_ref). Overlays of the copies iter11 made
(top iter12, bottom cover off):
- `evidence/exit_f0_iter12_vs_cover0.jpg`: exit f0 R backfilled with a hook that crosses the lane line.
- `evidence/a2d2a_f0_iter12_vs_cover0.jpg`: a2d2a f0 L and R backfilled with two diagonals forming an X over the bonnet.
iter12 leaves these frames empty, which is correct. Exit R −5.0 pts (1 of 20 frames) and a2d2a L −2.4 pts are within the
≤ 5 pt rule. With GPU SAM masks (the replay) exit stays 100/100 and a2d2a f0 R is gained.

## A bug this smoke caught (fixed before this run)
The first LaneAnchor also replaced edges the BEV tracker was coasting (held, no EdgeHold `hold_age`). The GPU replay
can't see this (iter11 did not dump that flag), but this smoke showed edge jit2 lane_split L 0.028 → 0.081, exit L
0.016 → 0.029, late R 0.021 → 0.035. LaneAnchor now treats tracker coasts as live; the GPU replay is unchanged by the fix.
iter12 dumps the flag (`edges_raw_held`) and `replay_full.py --held auto` uses it. On this smoke it narrows but does
not close the replay gap (late R jit2 smoke 0.026 / auto 0.025 / off 0.022; munich R 635 / 644 / 646). Stripe gores
(classical masks only) are not replayed.
