# iter13 SPEC (24 Nov 2026)

Base: iter12 (foundation-bev 6fc5234, GPU run /workspace/foundation_bev/iter12/). Entry `src/run_iter13.py`.
`patch.diff` = `git diff --no-index -M iter12_spec/src iter13_spec/src` (apply with `git apply -p2` inside an iter12 `src` tree).
No YOLO / TwinLiteNet in the pipeline (YOLOPv2 stays a separate baseline only).

## Changes (all new knobs default on unless noted; every one has an off switch)
1. **Parked-car clearance for synthetic edges** (`lines_mt.car_clear`, `edge_car_clear`, online and offline).
   Anchored / backfilled / EdgeHold edge points that fall inside a car footprint (WD3D box, + `FBEV_EDGE_CAR_TOL`=0.2 m)
   are pushed out to the kerb side of the car row as a parallel shift of the whole line (`FBEV_EDGE_CAR_PUSH=line`),
   limited to `FBEV_EDGE_CAR_W`=2.0 m; if no push clears it, the line is cut short 0.5 m before the car
   (`FBEV_EDGE_CAR_TRIM=1`, kept if ≥ `FBEV_EDGE_CAR_TRIM_MIN`=4 m remain), else dropped. `FBEV_EDGE_CAR_CLEAR=1`. Tracker coasts are not pushed
   (`FBEV_EDGE_CAR_COAST=0`: pushing them raised A2D2 edge wjit 0.062→0.095 / 0.056→0.081). Measured (kerb-pushed) edges
   are untouched.
2. **LaneAnchor rate limit** (`FBEV_EDGE_ANCHOR_RATE`=1.0 m/s): the anchored edge follows the lane as a parallel shift
   whose offset changes at most 1 m/s, so it cannot swing onto the car row when the lane wobbles. LaneAnchor now
   records the kerb push of the last measured edge (`note`) and keeps that offset.
3. **Fragment veto on tracker coasts** (`FBEV_EDGE_FRAG_COAST=1`): the iter12 diagonal-fragment veto also applies to
   tracker coasts (not EdgeHold coasts). Fills munich R f148-155 via the anchor; loses munich R f615 (a coasted diagonal).
4. **Anchor backfill** (`FBEV_EDGE_ANCHOR_BACKFILL=1`, offline): frames before the first LaneAnchor fit get the first
   anchored offset back-propagated (≤ 1.0 s), frame-0 cheap win (a2d2b f0 L edge).
5. **Low-fps time base** (`FBEV_LOWFPS_TIME=1`, `fr10(n, fps, floor)`): every frame-count window (StageLines birth/death,
   GoreTracker, DivergenceGore persist/hist, lane_min_hits, EdgeHold floor, back-fill min_hits, bev_gate, CLI
   coast/min_len/tail/hold/plane_every, n_agg) keeps its 10 Hz time span below 10 fps, floored (gore persistence 2
   frames = 0.5 s at 4 fps). **Bit-identical on all 9 public clips** (all ≥ 10 fps).
6. **Front-view occlusion** (`FBEV_DRAW_OCCLUDE=1`, display only): synthetic edges are not drawn over car boxes in the
   camera view (munich f200: the anchored R edge is ~1.3 m beyond the parked cars' outer flank but was painted over
   them). BEV and polylines unchanged.
7. **Private-clip plumbing** (fc1_merge, fc1_split): `run_iter13.py` / `wd3d_infer.py` / `yolo_baseline` merge
   `cams_private.json` / `motion_private.json` from FBEV_DS if present and skip the private tags otherwise.
   No private calibration, frames or labels are in this spec or the repo.
8. **Tools**: `car_audit.py`, `replay_iter13.sh`, `overlay_runs.py --dets`. The private-clip preparation and GT-scoring
   tools stay local (not in this repo).

## Replay vs the iter12 GPU run (`replay/replay_vs_iter12.md`; gpu12 / rep12 = iter12 code replay / iter13)
- Fidelity: rep12 = gpu12 except replay noise (sample lane jit2 0.002→0.003/0.004).
- **Gores identical** (exit 21-26, munich 760-765). **idsw, accel identical. No presence drop.**
- munich R z10 862 → **871**: gained 148-150, 152, 154, 155, **562, 563, 570, 571**; lost 615 (coasted diagonal, now vetoed).
- munich edge pres R 95.8 → 96.8, wjit 0.041 → 0.037, jit2 R 0.023 → 0.020. a2d2b edge pres L 97.1 → 99.4 (f0 L edge).
- **Regressions (disclosed):** city edge wjit 0.037 → 0.054, jit2 L 0.021 → 0.023; a2d2a edge jit2 L 0.035 → 0.036.
  Cause: the city L backfill (f0-1) is pushed out of the parked-car row while the first live edges still run on it;
  `FBEV_EDGE_CAR_CLEAR=0` restores iter12.
- Frame 0 still incomplete: lane_split R lane, munich R edge, a2d2a L lane, a2d2b L/R lane (no lane evidence at f0).
- Lane_split lane wjit 0.131 unchanged (structural, as iter12).

### Car audit (`replay/car_audit_*.txt`, exact footprint test)
Synthetic edge frames inside a car footprint: **gpu12 16 → iter13 0** (city L0-1, lane_split R0, munich R13, anchored
R190, 250, 251, 541, 546-549, a2d2a L0-2, a2d2b L14). Munich f189-251 anchored run: all clear (f193: 6.9 m through the
boxes → 7.6 m clear). Known: kerb-pushed **measured** edges still overlap footprints (munich 63, a2d2a 44 frames).

Overlays (iter12 GPU vs iter13, car footprints): `replay/overlays13/` (munich f13, 148, 190, 193, **200**, 240, 250,
541, 547, 562, 570, 615; city f0-2; a2d2b f0, 14; lane_split f0).

## Private clips (fc1_merge, fc1_split)
Two private company clips are evaluated locally against private ground truth. Clips, labels, frames, calibration and
per-frame results are not in this repo. The GPU run skipped them (no private files in its dataset).

## Tests and smokes
- `tests/`: **82 passed**, normal and NumPy-1 sim (`PYTHONPATH=tests/np1_sim`). `test_iter13.py` (15 tests): fr10, StageLines
  counts, car_inside / clear / trim / drop / push_local, edge_car_clear, _synthetic, anchor rate limit, frag coast veto, anchor
  backfill, off switch, front-view occlusion.
- **Kernel smoke** (`FBEV_SMOKE=1 FBEV_MAXF=8`, NumPy-1 sim, 11 clips incl. the two private ones (local only)):
  RC 0, fails none, 11 clips version iter13 (`replay/kernel_smoke_tail.txt`, private-clip lines omitted).
- **CPU smoke** (9 public clips, iter12 GPU dets + classical stand-in masks, iter12 code vs iter13 code,
  `cpu_smoke/cpu_smoke_iter12_vs_iter13.md`): 18/18 rc 0. accel, idsw, lane presence and gore frames identical; edge
  presence exit R 95 → 100, a2d2b L 97.1 → 97.7, no drop (the trim fallback fixed lane_split R 100 → 93.3 seen without
  it). Edge jitter rises: exit R jit2 0.044 → 0.051, lane_split R 0.063 → 0.066 / wjit 0.128 → 0.137, a2d2a wjit
  0.084 → 0.090, city wjit 0.088 → 0.091; falls: a2d2a R jit2 0.052 → 0.043, munich, late.
- The trim fallback never fires in the GPU replay (replay table and car audit byte-identical with and without it).

## GPU Bot run
- Entry `src/run_iter13.py`. Dataset = this folder (`src/`, `cams_iter13.json` = iter12 cams, `yolo_baseline/`) plus the
  same 9 public clip mp4s / cams.json / motion_assumptions.json as iter12. Private clips run only where their local
  files exist (never in git or shared folders); the GPU run skipped them.
- Clips: all 11 (`FBEV_TAGS` unset). Full stage 1+2 (no `FBEV_REUSE_DETS`). **Defaults only.** YOLO baseline on.
- `FBEV_EXTRA_STILLS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,63,80,86,100,107,119,121,123,133,140,146,148,149,150,151,152,153,154,155,160,165,166,170,180,189,190,191,192,193,194,195,196,200,220,240,242,250,251,258,266,267,300,450,520,523,526,535,538,541,545,546,547,548,549,557,560,561,562,563,565,568,570,571,572,574,577,587,603,615,625,647,677,692,697,700,760,762,878,879,880,881,882,883,884,885,886`
- Expected T4 time ≈ iter12 + ~3 min (120 extra frames).
- Outputs `/workspace/foundation_bev/iter13/`; `metrics_vs_iter12.md` (in `results/iter13/`).
