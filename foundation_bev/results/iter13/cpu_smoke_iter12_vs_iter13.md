# iter13 CPU smoke vs iter12 code (same inputs)

Inputs: iter12 GPU WD3D dets (`iter12/dets/wd3d_*.npz`) → stage 2 on CPU with **classical bev_ipm masks as the SAM
stand-in** (`--smoke --classical-masks`), all 9 public clips, full length. `run_cpu_smoke.sh` (iter13 code) vs
`run_iter12_cpu.sh` (iter12_spec code). **18/18 rc=0.** Full table: `cmp12_13.md` / `cmp12_13.json`.
The stand-in masks are much noisier than SAM: these numbers check robustness and code paths, not quality.

- **accel, idsw, lane presence, gore frames: identical on all 9 clips.**
- Edge presence: exit R 95.0 → 100, a2d2b L 97.1 → 97.7, munich L 98.6 → 98.7; **no drop**. (Before the car-clear
  trim fallback, lane_split R fell 100 → 93.3: a backfilled f1-2 line no push clears was dropped; it is now cut short in
  front of the car.)
- Edge jit2 / wjit at z10 (iter12 → iter13): a2d2a R jit2 0.052 → 0.043, wjit 0.084 → 0.090; a2d2b L jit2 0.035 → 0.036,
  wjit 0.084 → 0.083; city wjit 0.088 → 0.091; exit R jit2 0.044 → 0.051 (the 5% newly present frames); late L
  0.034 → 0.033; lane_split R jit2 0.063 → 0.066, wjit 0.128 → 0.137; munich R 0.027 → 0.026, wjit 0.054 → 0.053.
  Lane jitter unchanged. The rises are the same mechanism as city in the GPU replay (synthetic edges pushed off the
  car row next to live edges on it).
