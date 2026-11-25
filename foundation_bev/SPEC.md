# iter14 SPEC (25 Nov 2026, IST)

Base: iter13 (foundation-bev 15a5362, GPU run of 24 Nov). Entry `src/run_iter14.py`.
`patch.diff` = `git diff --no-index -M iter13_spec/src iter14_spec/src` (apply with `git apply -p2` from the parent of an
iter13 `src/`; verified to reproduce this `src/` exactly, sha256 in `results/iter14/SHA256SUMS_src.txt`).
Models unchanged: WildDet3D, SAM 2.1 hiera-small, Depth Anything V2-Small. No YOLO / TwinLiteNet in the pipeline.

**No GPU run for iter14.** The defaults are designed to be byte-identical to iter13 on all 9 public clips: the night
gate never fires on them and the ego guard rejects the new speed source on all of them. This was verified with the real
stage-2 models on CPU (below). A GPU run would only confirm a no-op. iter14 changes only private night / 4 fps footage.

## Changes (defaults; each has an off switch)
1. **ego14: road-masked BEV phase ego tier** (ON; `FBEV_EGO_ROAD=0` = iter13 selection).
   - A second BEV phase correlator runs on the SAM road minus the WildDet3D vehicle boxes (+4 px), eroded 5x5 and
     feathered (sigma 6 px; a hard mask edge is near-static and pulls the phase to 0), z 7-40 m, min_resp 0.10.
   - Used only where the iter13 selection would fall back to the prior, and only if: valid >= 40 % of pairs,
     >= 5 valid pairs, no repeated frames, and 0.33-3x the prior.
2. **Night gate for the adaptive thin-bright threshold (TB_ADAPT) and the road vertical-plane gate (VPLANE)**
   (`FBEV_TB_ADAPT` / `FBEV_ROAD_VPLANE` = auto; 0 = off, 1 = always).
   - TB_ADAPT: top-hat thresholds max(25 / 15, 1.2 x p90 of the road top-hat).
   - VPLANE: drops road pixels whose DAv2 depth is < 0.8 x flat-ground depth and that lie > 2.5 m off-axis
     (a lamp-lit barrier face merged into the road).
   - Both run only if `bev_smooth.night_cast` holds: road median hue 10-40 (sodium) AND
     ((road S > 80 and the frame top 40 % median V < 25) OR (S > 105 and top V < 90)).
     Knobs `FBEV_NIGHT_S/TOPV/S2/TOPV2`.
   - `FBEV_GATE_LOG` (default `$FBEV_OUT/gate.jsonl` in run_iter14.py) logs every gate call with the clip tag;
     diagnostics only. Summarise with `tools/gate_summary.py`.
   - A draft gate (S > 70 + hue, no darkness test) fired on warm daylight asphalt (KITTI highway, A2D2) and changed
     4 of 8 public clips. It was replaced by this one.
3. CPU stage-1 support in `wd3d_infer.py` (CUDA stream / pin_memory shims on CPU only, `--per-frame-dir`, `--frames`)
   and `FBEV_STAGE_DUMP` per-stage dumps. No effect on GPU runs.
4. **Evaluated, default OFF:**
   - `FBEV_EDGE_CAR_BF=trim` (clip-start backfilled edges cut before a parked car instead of pushed);
   - `FBEV_GORE_SPATIAL=1` (one-frame lane-edge divergence wedge over z 8-40 m, confirmed over `FBEV_GORE_SPATIAL_S`=0.5 s).
   Results below.

## Public clips: real stage-2 models on CPU (`results/iter14/public_cpu_check.md`)
SAM 2.1 hiera-s + DAv2-S on CPU, all frames, iter12 GPU WildDet3D dets. iter14 defaults vs the same code with the
night gates off:
- **Gate on: 0 calls on all 9 clips.**
- Polylines (excluding the version string) and tracks are identical on the 8 clips with a gates-off reference.
  munich has no such reference; with 0 gate firings it is identical by construction.
- ego: no clip selects the new source. The guard rejects it on sample, a2d2a and a2d2b; the rest use bev_phase as in iter13.

## Private night / 4 fps clips (aggregate scores only; no data, frames or labels in this repo)
Two private 4 fps FC1 clips with GT (a night motorway split and a rainy-day merge). CPU real-model runs, including
CPU WildDet3D stage 1. iter13 code -> iter14:
- **Night split:** false gore frames 21 -> 0.
  - z10 ego lane L / R median error 0.93 / 0.72 -> 0.22 / 0.05 m; z10 left road edge 3.95 -> 1.12 m.
  - All lines z10: lanes 0.80 -> 0.20 m, edges 0.96 -> 0.86 m. z30: lanes 0.99 -> 0.54 m, edges 1.72 -> 0.63 m.
  - Spurious lane lines 135 -> 41. GT lane-boundary recall z10 0.81 -> 0.91.
  - Ego distance error -47.7 % -> +15.5 %.
  - Costs: z30 lane recall 0.81 -> 0.75, lane classification 0.84 -> 0.66.
  - The fork itself is still not flagged (0/6 GT frames).
- **Rain merge:** ego distance error +74.7 % -> +3.4 %. Lines are otherwise unchanged (the gate never fires);
  merge gores 0/26 GT frames, 0 false.

## Replay vs the iter13 GPU run (`replay/REPLAY.md`, `replay/replay_vs_iter13.md`)
- **iter14 defaults:** polylines byte-identical to the iter13-behaviour replay on all 9 clips. Gores exit 21-26 and
  munich 760-765 only. Car audit: 0 synthetic edges inside a car.
- **`FBEV_EDGE_CAR_BF=trim`:** fixes the iter13 city regression (edge wjit 0.054 -> 0.037, jit2 L 0.023 -> 0.021;
  car audit 0), but z20 L presence drops (city 100 -> 93.3, clip-start frames; a2d2a 83.7 -> 83.1). Default off.
- **Frame-0 completeness:** no cheap fix.
  - a2d2a / a2d2b: no lane evidence for 4-16 s.
  - lane_split R: the split branch.
  - munich R edge: iter12 diagonal-copy veto.
- **`FBEV_GORE_SPATIAL=1`:** keeps the exit and munich gores, but adds false gores on late, munich, a2d2a and a2d2b
  and on the private split. Not adopted.

## Tests and smokes
- `tests/`: 98 passed, both normal and with the NumPy-1 sim. `test_iter14.py` (16): night gate on / off, TB / VPLANE
  daylight no-op, gate log, backfill trim knob, spatial wedge, ego14 default.
- Kernel smoke (`FBEV_SMOKE=1 FBEV_MAXF=8`, NumPy-1 sim, 9 public clips): RC 0, fails none (`replay/kernel_smoke_tail.txt`;
  with only 8 frames exit_approach takes bev_road 22.8 vs prior 22; full-length runs use bev_phase).
