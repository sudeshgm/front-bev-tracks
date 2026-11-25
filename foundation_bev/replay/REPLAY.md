# iter14 replay vs the iter13 GPU run (post-tracker; CPU, no weights)

Script: `tools/replay_iter14.sh` (RUN=/workspace/foundation_bev/iter13). Table: `replay_vs_iter13.md` / `.json`.
Columns: gpu13 (GPU output) | rep13 (iter14 src, `FBEV_EDGE_CAR_BF=push` = iter13 behaviour) | iter14 (defaults) |
bf_trim (`FBEV_EDGE_CAR_BF=trim`) | gore_spatial (`FBEV_GORE_SPATIAL=1`).
Stage-2 changes (ego14, the night gate) are NOT replayable here; they are verified by the real-model CPU runs (SPEC.md).

- **iter14 defaults:** polylines byte-identical to rep13 on all 9 clips. rep13 = gpu13 except replay noise
  (sample lane jit2 .002 -> .003/.004). Gores exit 21-26 and munich 760-765 only. Car audit: 0 synthetic edges inside a car
  (`car_audit_iter14.txt`).
- **bf_trim:**
  - city edge wjit .054 -> **.037**, jit2 L .023 -> .021, car audit 0;
  - a2d2a L20 jit .088 -> .083;
  - costs: z20 L presence city 100 -> 93.3, a2d2a 83.7 -> 83.1. Off by default.
  - `local` variant (`bfl/`): city wjit .054 -> .076 (rejected).
- **gore_spatial:**
  - adds false gores on late f11-17, munich f575-581, a2d2a f22-26 + f38-42, a2d2b f113-123; exit / munich true gores kept;
  - rejected (default off).
- **Frame 0 (`f0 laneL,R edgeL,R`):** unchanged. Causes:
  - lane_split R = split branch at f7;
  - munich R edge diagonal-copy veto;
  - a2d2a / a2d2b no lane evidence before f38 / f48 / f162.
  - `lb8/`: LANE_BACKFILL_S 0.8 = identical.
