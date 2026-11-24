# Metrics versus iter11 (9-clip iter12)

iter11 is the completed GPU run at `/workspace/foundation_bev/iter11/`.
iter12 is this full stage 1+2 GPU run (`/workspace/foundation_bev/iter12/`, pulled from volume `gpubot-bev-iter12`). Both scored KITTI and A2D2 at capture_hz (10 fps) and Munich at 30 fps, taken from each clip's polylines JSON and `metrics.json`.
Gore frames are counts of frames with a non-empty `gores` list; sources are `stripes` / `fork` / `widen`.
Held-edge / kerb_occ / frag / backfilled / anchored counts are frames with at least one edge that has the corresponding flag true (`held`, `kerb_occ`, `frag`, `backfill`, `anchored`).
Knobs were left at code defaults, including the new iter12 defaults (`FBEV_EDGE_BACKFILL_MODE=hold`, `FBEV_EDGE_BACKFILL_COVER=1`, `FBEV_EDGE_BACKFILL_S=1.0`, `FBEV_EDGE_ANCHOR_S=3.0`, `FBEV_EDGE_ANCHOR_DEV=0.15`, `FBEV_KERB_TRAFFIC=outer`). They were not changed after this comparison.
Caveat (SPEC): stage 1 is not deterministic between runs, so clip accel can move by about ±0.05 independently of this code. iter12 does not touch tracks.

| clip | accel iter11 → iter12 | n_tracks 11→12 | idsw 11→12 | lane L10/R10 pres % | lane L10/R10 jit2 | lane wjit | edge L10/R10 pres % | edge L10/R10 jit2 | edge wjit | gore frames iter11 → iter12 | held-edge 11→12 | kerb_occ 11→12 | frag | backfilled | anchored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample | 0.095 → 0.095 | 1→1 | 0.0→0.0 | 100.0/100.0 → 100.0/100.0 | 0.002/0.002 → 0.002/0.002 | 0.001 → 0.001 | 100.0/100.0 → 100.0/100.0 | 0.020/0.009 → 0.020/0.009 | 0.022 → 0.019 | 0 → 0 | 0→0 | 0→0 | 0 (—) | 2 (f0000–f0001) | 0 (—) |
| exit_approach | 0.546 → 0.546 | 2→2 | 0.0→0.0 | 100.0/100.0 → 100.0/100.0 | 0.005/0.008 → 0.005/0.008 | 0.007 → 0.007 | 100.0/100.0 → 100.0/100.0 | 0.016/0.023 → 0.016/0.024 | 0.047 → 0.042 | 6 (widen 6; f0021–f0026) → 6 (widen 6; f0021–f0026) | 0→0 | 0→0 | 0 (—) | 2 (f0000–f0001) | 0 (—) |
| late | 0.534 → 0.534 | 5→5 | 0.0→0.0 | 97.5/100.0 → 97.5/100.0 | 0.011/0.012 → 0.011/0.012 | 0.006 → 0.006 | 100.0/100.0 → 100.0/100.0 | 0.020/0.014 → 0.020/0.015 | 0.041 → 0.034 | 0 → 0 | 9→9 | 0→0 | 0 (—) | 5 (f0000–f0004) | 0 (—) |
| final | 0.574 → 0.574 | 3→3 | 0.0→0.0 | 100.0/100.0 → 100.0/100.0 | 0.014/0.004 → 0.014/0.004 | 0.006 → 0.006 | 100.0/100.0 → 100.0/100.0 | 0.029/0.021 → 0.029/0.023 | 0.026 → 0.022 | 0 → 0 | 0→0 | 0→0 | 0 (—) | 4 (f0000–f0003) | 0 (—) |
| city | 0.548 → 0.548 | 5→5 | 0.0→0.0 | 100.0/100.0 → 100.0/100.0 | 0.002/0.001 → 0.002/0.001 | 0.001 → 0.001 | 100.0/100.0 → 100.0/100.0 | 0.018/0.010 → 0.021/0.010 | 0.076 → 0.037 | 0 → 0 | 2→2 | 0→0 | 0 (—) | 2 (f0000–f0001) | 0 (—) |
| lane_split | 0.951 → 0.951 | 13→13 | 0.0→0.0 | 100.0/73.3 → 100.0/73.3 | 0.011/0.022 → 0.011/0.022 | 0.131 → 0.131 | 100.0/100.0 → 100.0/100.0 | 0.042/0.041 → 0.042/0.042 | 0.115 → 0.115 | 0 → 0 | 7→7 | 15→15 | 0 (—) | 2 (f0000–f0001) | 0 (—) |
| munich | 0.655 → 0.655 | 87→87 | 0.0→0.0 | 85.9/65.4 → 85.9/65.4 | 0.003/0.002 → 0.003/0.002 | 0.006 → 0.006 | 100.0/86.3 → 100.0/95.8 | 0.010/0.021 → 0.010/0.023 | 0.041 → 0.041 | 6 (fork 6; f0760–f0765) → 6 (fork 6; f0760–f0765) | 221→286 | 341→339 | 0 (—) | 3 (f0000–f0001,f0013) | 119 (f0144–f0147,f0165–f0166,f0189–f0251,f0520–f0539,f0541,f0546–f0561,f0564–f0569,f0576–f0577,f0656–f0657,f0664–f0665,f0704) |
| a2d2a | 0.495 → 0.495 | 23→23 | 0.0→0.0 | 15.1/26.7 → 15.1/26.7 | 0.008/0.007 → 0.008/0.007 | — → — | 97.7/94.8 → 97.7/98.8 | 0.036/0.030 → 0.035/0.030 | 0.065 → 0.062 | 0 → 0 | 34→34 | 63→63 | 0 (—) | 7 (f0000–f0006) | 0 (—) |
| a2d2b | 0.821 → 0.821 | 11→11 | 0.0→0.0 | 8.1/2.3 → 8.1/2.3 | 0.012/0.035 → 0.012/0.035 | — → — | 97.1/100.0 → 97.1/100.0 | 0.038/0.032 → 0.038/0.032 | 0.057 → 0.056 | 0 → 0 | 39→39 | 0→0 | 0 (—) | 5 (f0000–f0004) | 1 (f0014) |

## Gore check (must be ONLY exit_approach f21–26 and munich f760–765)

**PASS.** Gores fire only on exit_approach f0021–f0026 (widen) and munich f0760–f0765 (fork). No other clip fires.

- sample: 0
- exit_approach: 6 (widen 6; f0021–f0026)
- late: 0
- final: 0
- city: 0
- lane_split: 0
- munich: 6 (fork 6; f0760–f0765)
- a2d2a: 0
- a2d2b: 0

## Presence drops > 5 points vs iter11, accel rise > 0.05, or idsw > 0 (flag only; knobs unchanged)

A drop is flagged when iter12 lane or edge L10/R10 presence is more than 5 points below iter11, per clip and side. Accel is flagged when iter12 `accel_rms` is more than 0.05 above iter11. idsw is flagged when the proxy is above 0.

**No presence drop > 5 points** versus iter11.

**No accel rise above 0.05** versus iter11. Clip-level `accel_rms` and `n_tracks` are **unchanged** on all 9 clips.

**idsw is 0** on all 9 clips in iter12 and iter11.

Notable non-flag movements (for context, not flags):
- sample edge wjit: iter11 0.022 → iter12 0.019 (Δ -0.003).
- exit_approach edge wjit: iter11 0.047 → iter12 0.042 (Δ -0.005).
- late edge wjit: iter11 0.041 → iter12 0.034 (Δ -0.007).
- final edge wjit: iter11 0.026 → iter12 0.022 (Δ -0.004).
- city edge wjit: iter11 0.076 → iter12 0.037 (Δ -0.039).
- munich edge R10 pres: iter11 86.3 → iter12 95.8 (Δ +9.5).
- a2d2a edge R10 pres: iter11 94.8 → iter12 98.8 (Δ +4.0).
- a2d2a edge wjit: iter11 0.065 → iter12 0.062 (Δ -0.003).

## Held-edge, kerb_occ, frag, backfilled-edge and anchored-edge frames per clip

| clip | held iter11 → iter12 (ranges) | kerb_occ iter11 → iter12 (n_kerb_edges, ranges) | frag (n_edges, ranges) | backfilled (n_edges, ranges) | anchored (n_edges, ranges) |
| --- | --- | --- | --- | --- | --- |
| sample | 0 → 0 (—) | 0 → 0 (0; —) | 0 (0; —) | 2 (4; f0000–f0001) | 0 (0; —) |
| exit_approach | 0 → 0 (—) | 0 → 0 (0; —) | 0 (0; —) | 2 (4; f0000–f0001) | 0 (0; —) |
| late | 9 → 9 (f0014,f0021–f0022,f0033–f0036,f0038–f0039) | 0 → 0 (0; —) | 0 (0; —) | 5 (7; f0000–f0004) | 0 (0; —) |
| final | 0 → 0 (—) | 0 → 0 (0; —) | 0 (0; —) | 4 (7; f0000–f0003) | 0 (0; —) |
| city | 2 → 2 (f0028–f0029) | 0 → 0 (0; —) | 0 (0; —) | 2 (4; f0000–f0001) | 0 (0; —) |
| lane_split | 7 → 7 (f0005–f0006,f0018–f0019,f0025,f0028–f0029) | 15 → 15 (15; f0015–f0029) | 0 (0; —) | 2 (4; f0000–f0001) | 0 (0; —) |
| munich | 221 → 286 (f0004–f0007,f0009–f0013,f0044,f0054–f0055,f0062–f0067,f0120–f0121,f0136–f0141,f0144–f0147,f0152–f0153,f0156–f0157,f0160–f0166,f0170–f0251,f0253–f0254,f0259–f0260,f0264–f0265,f0290–f0293,f0298–f0301,f0336–f0337,f0360–f0361,f0414–f0415,f0422–f0423,f0432–f0433,f0436–f0437,f0456–f0457,f0474–f0475,f0512–f0513,f0515–f0516,f0518–f0539,f0541,f0544–f0571,f0573–f0577,f0584–f0587,f0596–f0611,f0615–f0617,f0622–f0623,f0641–f0643,f0648–f0657,f0660–f0665,f0668–f0671,f0688–f0689,f0692–f0693,f0700–f0704,f0706–f0707,f0858–f0859,f0866–f0867,f0870–f0873,f0876–f0877,f0896–f0899) | 341 → 339 (339; f0004–f0082,f0152–f0164,f0167–f0188,f0252–f0255,f0276–f0298,f0300–f0430,f0448–f0470,f0482–f0484,f0732–f0734,f0762–f0764,f0805,f0830–f0832,f0866–f0890,f0894–f0899) | 0 (0; —) | 3 (3; f0000–f0001,f0013) | 119 (119; f0144–f0147,f0165–f0166,f0189–f0251,f0520–f0539,f0541,f0546–f0561,f0564–f0569,f0576–f0577,f0656–f0657,f0664–f0665,f0704) |
| a2d2a | 34 → 34 (f0004–f0006,f0008–f0009,f0011–f0013,f0015–f0017,f0020,f0022,f0024,f0027–f0028,f0059,f0062,f0067–f0068,f0092–f0095,f0108–f0111,f0125,f0130,f0138–f0139,f0144,f0146) | 63 → 63 (84; f0015,f0053–f0113,f0155) | 0 (0; —) | 7 (10; f0000–f0006) | 0 (0; —) |
| a2d2b | 39 → 39 (f0007–f0008,f0010–f0015,f0038–f0040,f0068–f0071,f0082–f0083,f0088–f0089,f0091,f0101–f0104,f0107–f0108,f0126–f0129,f0143–f0151) | 0 → 0 (0; —) | 0 (0; —) | 5 (6; f0000–f0004) | 1 (1; f0014) |

## Munich right edge at z = 10: trimmed `edges_bev` vs `edges_raw` (vs iter11's 777)

Presence uses the same definition as `edges_R10_pres` in `smoothness_metrics.py`: `x_at(z=10)` on `edges_bev` (nearest right edge) for the trimmed output, and `x_at(z=10)` on `edges_raw.main_R` for the untrimmed main right edge. Iter11 was **777 / 900** present (86.3%). SPEC expected about **860 / 900**, with f165–166 and f561–571 present and f572–575 absent.

- Trimmed `edges_bev` right edge present at z = 10: **862** / 900 (**95.8%**, metrics `edges_R10_pres` 95.8). Missing: **38**.
- Untrimmed `edges_raw.main_R` present at z = 10: **824** / 900 (91.6%). Raw starts beyond z 10: 8 (f0148–f0155). Raw absent: 2.
- Change vs iter11: **777 → 862 (Δ +85)**, in line with the expected about 860 / 900 (Δ vs 860: +2).
- **Gained vs iter11 (85 frames):** f0013,f0144–f0147,f0165–f0166,f0189–f0251,f0535–f0539,f0541,f0561,f0564–f0569,f0576–f0577.
- **Lost vs iter11 (0 frames):** —.

SPEC frame checks:
- **PASS** f165–166 (anchored, expected present): present f0165–f0166; absent —.
- **FAIL** f561–571 (anchored, expected present): present f0561,f0564–f0569; absent f0562–f0563,f0570–f0571.
- **PASS** f572–575 (expected absent): absent f0572–f0575; unexpectedly present —.

Of the 38 iter12 trimmed misses at z = 10: **28** still have raw `main_R` at z = 10 (trimmed away), **8** have raw starting beyond z 10, **2** have no raw `main_R`.

## Frame 0: lanes and road edges present

Presence at z = 10 using the same `x_at` / `side_series` definition as the L10/R10 metrics, on frame 0 of every clip. SPEC expected the **a2d2a right edge** to be present at z = 10.

| clip | recorded frame | lane L10 | lane R10 | edge L10 | edge R10 | n_lanes | n_edges |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sample | f0000 | yes | yes | yes | yes | 2 | 2 |
| exit_approach | f0000 | yes | yes | yes | yes | 2 | 2 |
| late | f0000 | yes | yes | yes | yes | 2 | 2 |
| final | f0000 | yes | yes | yes | yes | 2 | 2 |
| city | f0000 | yes | yes | yes | yes | 4 | 2 |
| lane_split | f0000 | yes | NO | yes | yes | 1 | 2 |
| munich | f0000 | yes | yes | yes | NO | 2 | 1 |
| a2d2a | f0000 | NO | yes | yes | yes | 1 | 2 |
| a2d2b | f0000 | NO | NO | NO | yes | 0 | 1 |

**a2d2a right edge at z = 10 on frame 0: PASS** (present=True).

Notes:
- **sample:** both lanes and both road edges present at z = 10.
- **exit_approach:** both lanes and both road edges present at z = 10.
- **late:** both lanes and both road edges present at z = 10.
- **final:** both lanes and both road edges present at z = 10.
- **city:** both lanes and both road edges present at z = 10.
- **lane_split:** missing at z = 10: lane R (n_lanes=1, n_edges=2).
- **munich:** missing at z = 10: edge R (n_lanes=2, n_edges=1).
- **a2d2a:** missing at z = 10: lane L (n_lanes=1, n_edges=2).
- **a2d2b:** missing at z = 10: lane L, lane R, edge L (n_lanes=0, n_edges=1).

## Lane presence and width jitter vs YOLO (KITTI clips)

YOLOPv2 is the measurement baseline only (no road edges). Numbers are L10/R10 presence (%), lane wjit, and lane jit2 L/R at z = 10, from `metrics_yolo_vs_run.json` (YOLO) and this run's `metrics.json` (iter11 → iter12).

| clip | lane pres L/R YOLO · iter11 → iter12 | lane wjit YOLO · iter11 → iter12 | lane jit2 L YOLO · iter11 → iter12 | lane jit2 R YOLO · iter11 → iter12 |
| --- | --- | --- | --- | --- |
| sample | 100.0/100.0 · 100.0/100.0 → 100.0/100.0 | 0.006 · 0.001 → 0.001 | 0.019 · 0.002 → 0.002 | 0.005 · 0.002 → 0.002 |
| exit | 100.0/100.0 · 100.0/100.0 → 100.0/100.0 | 0.007 · 0.007 → 0.007 | 0.013 · 0.005 → 0.005 | 0.009 · 0.008 → 0.008 |
| late | 100.0/100.0 · 97.5/100.0 → 97.5/100.0 | 0.009 · 0.006 → 0.006 | 0.014 · 0.011 → 0.011 | 0.014 · 0.012 → 0.012 |
| final | 100.0/100.0 · 100.0/100.0 → 100.0/100.0 | 0.008 · 0.006 → 0.006 | 0.024 · 0.014 → 0.014 | 0.017 · 0.004 → 0.004 |
| city | 100.0/100.0 · 100.0/100.0 → 100.0/100.0 | 0.005 · 0.001 → 0.001 | 0.001 · 0.002 → 0.002 | 0.004 · 0.001 → 0.001 |
| lane_split | 76.7/26.7 · 100.0/73.3 → 100.0/73.3 | 0.092 · 0.131 → 0.131 | 0.257 · 0.011 → 0.011 | 0.721 · 0.022 → 0.022 |

Versus YOLO on this GPU run:
- **sample:** pres L/R 100.0/100.0 (YOLO 100.0/100.0); wjit 0.001 (YOLO 0.006); jit2 L/R 0.002/0.002 (YOLO 0.019/0.005).
- **exit:** pres L/R 100.0/100.0 (YOLO 100.0/100.0); wjit 0.007 (YOLO 0.007); jit2 L/R 0.005/0.008 (YOLO 0.013/0.009).
- **late:** pres L/R 97.5/100.0 (YOLO 100.0/100.0); wjit 0.006 (YOLO 0.009); jit2 L/R 0.011/0.012 (YOLO 0.014/0.014).
- **final:** pres L/R 100.0/100.0 (YOLO 100.0/100.0); wjit 0.006 (YOLO 0.008); jit2 L/R 0.014/0.004 (YOLO 0.024/0.017).
- **city:** pres L/R 100.0/100.0 (YOLO 100.0/100.0); wjit 0.001 (YOLO 0.005); jit2 L/R 0.002/0.001 (YOLO 0.001/0.004).
- **lane_split:** pres L/R 100.0/73.3 (YOLO 76.7/26.7); wjit 0.131 (YOLO 0.092); jit2 L/R 0.011/0.022 (YOLO 0.257/0.721).

