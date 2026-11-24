# Metrics versus iter10 (9-clip iter11)

iter10 is the completed GPU run at `/workspace/foundation_bev/iter10/`.
iter11 is this full stage 1+2 GPU run (`/workspace/foundation_bev/iter11/`, pulled from volume `gpubot-bev-iter11`). Both scored KITTI and A2D2 at capture_hz (10 fps) and Munich at 30 fps, taken from each clip's polylines JSON and `metrics.json`.
Gore frames are counts of frames with a non-empty `gores` list; sources are `stripes` / `fork` / `widen`.
Held-edge counts are the number of frames with at least one edge that has `held: true`.
kerb_occ counts are the number of frames with at least one edge that has `kerb_occ: true`.
frag counts are frames with at least one edge that has `frag: true`.
backfilled-edge counts are frames with at least one edge that has `backfill: true`.
Knobs were left at code defaults, including the new iter11 defaults (`FBEV_GORE_KERB_RESET=1`, `FBEV_EDGE_FRAG_M=6.0`, `FBEV_EDGE_FRAG_DEV=0.25`, `FBEV_EDGE_FRAG_EXT_M=12`, `FBEV_LANE_BACKFILL_S=0.6`, `FBEV_EDGE_BACKFILL_CV=1`, `FBEV_PAIR_W_MODE=kf`, `FBEV_PAIR_KF_MODEL=rw`, `FBEV_PAIR_KF_Q=0.1`, `FBEV_PAIR_KF_R=0.15`, `FBEV_PAIR_KF_MIN_S=1.0`). They were not changed after this comparison.
Caveat (SPEC): stage 1 is not deterministic between runs, so clip accel can move by about ±0.05 independently of this code. iter11 does not touch tracks.

| clip | accel iter10 → iter11 | n_tracks 10→11 | idsw 10→11 | lane L10/R10 pres % | lane L10/R10 jit2 | lane wjit | edge L10/R10 pres % | edge L10/R10 jit2 | edge wjit | gore frames iter10 → iter11 | held-edge frames 10→11 | kerb_occ frames 10→11 | frag frames | backfilled-edge frames |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample | 0.095 → 0.095 | 1→1 | 0.0→0.0 | 100.0/100.0 → 100.0/100.0 | 0.004/0.003 → 0.002/0.002 | 0.004 → 0.001 | 95.0/95.0 → 100.0/100.0 | 0.020/0.009 → 0.020/0.009 | 0.019 → 0.022 | 0 → 0 | 0→0 | 0→0 | 0 (—) | 2 (f0000–f0001) |
| exit_approach | 0.546 → 0.546 | 2→2 | 0.0→0.0 | 100.0/100.0 → 100.0/100.0 | 0.005/0.014 → 0.005/0.008 | 0.017 → 0.007 | 95.0/95.0 → 100.0/100.0 | 0.016/0.024 → 0.016/0.023 | 0.048 → 0.047 | 6 (widen 6; f0021–f0026) → 6 (widen 6; f0021–f0026) | 0→0 | 0→0 | 0 (—) | 2 (f0000–f0001) |
| late | 0.534 → 0.534 | 5→5 | 0.0→0.0 | 97.5/100.0 → 97.5/100.0 | 0.011/0.007 → 0.011/0.012 | 0.019 → 0.006 | 95.0/87.5 → 100.0/100.0 | 0.020/0.015 → 0.020/0.014 | 0.035 → 0.041 | 0 → 0 | 9→9 | 0→0 | 0 (—) | 5 (f0000–f0004) |
| final | 0.574 → 0.574 | 3→3 | 0.0→0.0 | 93.3/90.0 → 100.0/100.0 | 0.016/0.004 → 0.014/0.004 | 0.012 → 0.006 | 90.0/86.7 → 100.0/100.0 | 0.029/0.025 → 0.029/0.021 | 0.023 → 0.026 | 0 → 0 | 0→0 | 0→0 | 0 (—) | 4 (f0000–f0003) |
| city | 0.548 → 0.548 | 5→5 | 0.0→0.0 | 100.0/100.0 → 100.0/100.0 | 0.002/0.003 → 0.002/0.001 | 0.004 → 0.001 | 93.3/93.3 → 100.0/100.0 | 0.021/0.010 → 0.018/0.010 | 0.054 → 0.076 | 0 → 0 | 2→2 | 0→0 | 0 (—) | 2 (f0000–f0001) |
| lane_split | 0.951 → 0.951 | 13→13 | 0.0→0.0 | 100.0/73.3 → 100.0/73.3 | 0.011/0.022 → 0.011/0.022 | 0.131 → 0.131 | 93.3/93.3 → 100.0/100.0 | 0.042/0.042 → 0.042/0.041 | 0.127 → 0.115 | 0 → 0 | 7→7 | 15→15 | 0 (—) | 2 (f0000–f0001) |
| munich | 0.655 → 0.655 | 87→87 | 0.0→0.0 | 85.9/65.4 → 85.9/65.4 | 0.003/0.002 → 0.003/0.002 | 0.006 → 0.006 | 99.8/86.2 → 100.0/86.3 | 0.010/0.022 → 0.010/0.021 | 0.040 → 0.041 | 11 (fork 6, widen 5; f0760–f0765,f0881–f0885) → 6 (fork 6; f0760–f0765) | 189→221 | 341→341 | 0 (—) | 2 (f0000–f0001) |
| a2d2a | 0.495 → 0.495 | 23→23 | 0.0→0.0 | 15.1/26.7 → 15.1/26.7 | 0.008/0.007 → 0.008/0.007 | — → — | 95.9/94.8 → 97.7/94.8 | 0.036/0.030 → 0.036/0.030 | 0.065 → 0.065 | 0 → 0 | 34→34 | 63→63 | 0 (—) | 3 (f0000–f0002) |
| a2d2b | 0.821 → 0.821 | 11→11 | 0.0→0.0 | 8.1/2.3 → 8.1/2.3 | 0.012/0.035 → 0.012/0.035 | — → — | 97.1/97.1 → 97.1/100.0 | 0.038/0.032 → 0.038/0.032 | 0.057 → 0.057 | 0 → 0 | 39→39 | 0→0 | 0 (—) | 6 (f0000–f0005) |

## Gore check (must be ONLY exit_approach f21–26 and munich f760–765; munich f881–885 must be gone)

**PASS.** Gores fire only on exit_approach f0021–f0026 (widen) and munich f0760–f0765 (fork). Munich f0881–f0885 is gone. No other clip fires.

- sample: 0
- exit_approach: 6 (widen 6; f0021–f0026)
- late: 0
- final: 0
- city: 0
- lane_split: 0
- munich: 6 (fork 6; f0760–f0765)
- a2d2a: 0
- a2d2b: 0

## Presence drops > 5 points vs iter10, accel rise > 0.05, or idsw > 0 (flag only; knobs unchanged)

A drop is flagged when iter11 lane or edge L10/R10 presence is more than 5 points below iter10, per clip and side. Accel is flagged when iter11 `accel_rms` is more than 0.05 above iter10. idsw is flagged when the proxy is above 0.

**No presence drop > 5 points** versus iter10.

**No accel rise above 0.05** versus iter10. Clip-level `accel_rms` and `n_tracks` are **unchanged** on all 9 clips.

**idsw is 0** on all 9 clips in iter11 and iter10.

Notable non-flag movements (for context, not flags):
- sample edge L10 pres: iter10 95.0 → iter11 100.0 (Δ +5.0).
- sample edge R10 pres: iter10 95.0 → iter11 100.0 (Δ +5.0).
- sample lane wjit: iter10 0.004 → iter11 0.001 (Δ -0.003).
- sample lane L jit2: iter10 0.004 → iter11 0.002 (Δ -0.002).
- sample edge wjit: iter10 0.019 → iter11 0.022 (Δ +0.003).
- exit_approach edge L10 pres: iter10 95.0 → iter11 100.0 (Δ +5.0).
- exit_approach edge R10 pres: iter10 95.0 → iter11 100.0 (Δ +5.0).
- exit_approach lane wjit: iter10 0.017 → iter11 0.007 (Δ -0.010).
- exit_approach lane R jit2: iter10 0.014 → iter11 0.008 (Δ -0.006).
- late edge L10 pres: iter10 95.0 → iter11 100.0 (Δ +5.0).
- late edge R10 pres: iter10 87.5 → iter11 100.0 (Δ +12.5).
- late lane wjit: iter10 0.019 → iter11 0.006 (Δ -0.013).
- late lane R jit2: iter10 0.007 → iter11 0.012 (Δ +0.005).
- late edge wjit: iter10 0.035 → iter11 0.041 (Δ +0.006).
- final lane L10 pres: iter10 93.3 → iter11 100.0 (Δ +6.7).
- final lane R10 pres: iter10 90.0 → iter11 100.0 (Δ +10.0).
- final edge L10 pres: iter10 90.0 → iter11 100.0 (Δ +10.0).
- final edge R10 pres: iter10 86.7 → iter11 100.0 (Δ +13.3).
- final lane wjit: iter10 0.012 → iter11 0.006 (Δ -0.006).
- final lane L jit2: iter10 0.016 → iter11 0.014 (Δ -0.002).
- final edge wjit: iter10 0.023 → iter11 0.026 (Δ +0.003).
- city edge L10 pres: iter10 93.3 → iter11 100.0 (Δ +6.7).
- city edge R10 pres: iter10 93.3 → iter11 100.0 (Δ +6.7).
- city lane wjit: iter10 0.004 → iter11 0.001 (Δ -0.003).
- city lane R jit2: iter10 0.003 → iter11 0.001 (Δ -0.002).
- city edge wjit: iter10 0.054 → iter11 0.076 (Δ +0.022).
- lane_split edge L10 pres: iter10 93.3 → iter11 100.0 (Δ +6.7).
- lane_split edge R10 pres: iter10 93.3 → iter11 100.0 (Δ +6.7).
- lane_split edge wjit: iter10 0.127 → iter11 0.115 (Δ -0.012).
- a2d2a edge L10 pres: iter10 95.9 → iter11 97.7 (Δ +1.8).
- a2d2b edge R10 pres: iter10 97.1 → iter11 100.0 (Δ +2.9).

## Held-edge, kerb_occ, frag and backfilled-edge frames per clip

| clip | held iter10 → iter11 (ranges) | kerb_occ iter10 → iter11 (n_kerb_edges, ranges) | frag frames (n_edges, ranges) | backfilled-edge frames (n_edges, ranges) |
| --- | --- | --- | --- | --- |
| sample | 0 → 0 (—) | 0 → 0 (0; —) | 0 (0; —) | 2 (4; f0000–f0001) |
| exit_approach | 0 → 0 (—) | 0 → 0 (0; —) | 0 (0; —) | 2 (4; f0000–f0001) |
| late | 9 → 9 (f0014,f0021–f0022,f0033–f0036,f0038–f0039) | 0 → 0 (0; —) | 0 (0; —) | 5 (7; f0000–f0004) |
| final | 0 → 0 (—) | 0 → 0 (0; —) | 0 (0; —) | 4 (7; f0000–f0003) |
| city | 2 → 2 (f0028–f0029) | 0 → 0 (0; —) | 0 (0; —) | 2 (4; f0000–f0001) |
| lane_split | 7 → 7 (f0005–f0006,f0018–f0019,f0025,f0028–f0029) | 15 → 15 (15; f0015–f0029) | 0 (0; —) | 2 (4; f0000–f0001) |
| munich | 189 → 221 (f0004–f0007,f0009–f0013,f0044,f0054–f0055,f0062–f0067,f0120–f0121,f0136–f0141,f0144–f0147,f0152–f0153,f0156–f0157,f0160–f0166,f0170–f0188,f0202–f0203,f0246–f0247,f0249,f0253–f0254,f0259–f0260,f0264–f0265,f0290–f0293,f0298–f0301,f0336–f0337,f0360–f0361,f0414–f0415,f0422–f0423,f0432–f0433,f0436–f0437,f0456–f0457,f0474–f0475,f0512–f0513,f0515–f0516,f0518–f0535,f0541,f0544–f0560,f0562–f0567,f0570–f0571,f0573–f0577,f0584–f0587,f0596–f0611,f0615–f0617,f0622–f0623,f0641–f0643,f0648–f0657,f0660–f0665,f0668–f0671,f0688–f0689,f0692–f0693,f0700–f0704,f0706–f0707,f0858–f0859,f0866–f0867,f0870–f0873,f0876–f0877,f0896–f0899) | 341 → 341 (341; f0004–f0082,f0152–f0188,f0252–f0255,f0276–f0298,f0300–f0430,f0448–f0470,f0482–f0484,f0732–f0734,f0762–f0764,f0805,f0830–f0832,f0866–f0890,f0894–f0899) | 0 (0; —) | 2 (4; f0000–f0001) |
| a2d2a | 34 → 34 (f0004–f0006,f0008–f0009,f0011–f0013,f0015–f0017,f0020,f0022,f0024,f0027–f0028,f0059,f0062,f0067–f0068,f0092–f0095,f0108–f0111,f0125,f0130,f0138–f0139,f0144,f0146) | 63 → 63 (84; f0015,f0053–f0113,f0155) | 0 (0; —) | 3 (3; f0000–f0002) |
| a2d2b | 39 → 39 (f0007–f0008,f0010–f0015,f0038–f0040,f0068–f0071,f0082–f0083,f0088–f0089,f0091,f0101–f0104,f0107–f0108,f0126–f0129,f0143–f0151) | 0 → 0 (0; —) | 0 (0; —) | 6 (11; f0000–f0005) |

## Munich right edge at z = 10: trimmed `edges_bev` vs `edges_raw` (vs iter10's 776)

Presence uses the same definition as `edges_R10_pres` in `smoothness_metrics.py`: `x_at(z=10)` on `edges_bev` (nearest right edge) for the trimmed output, and `x_at(z=10)` on `edges_raw.main_R` for the untrimmed main right edge. Iter10 was **776 / 900** present (86.2%).

- Trimmed `edges_bev` right edge present at z = 10: **777** / 900 (**86.3%**, metrics `edges_R10_pres` 86.3). Missing: **123**.
- Untrimmed `edges_raw.main_R` present at z = 10: **824** / 900 (91.6%). Raw starts beyond z 10: 8 (f0148–f0155). Raw absent: 44.
- Change vs iter10: **776 → 777 (Δ +1)**.
- **Gained vs iter10 (19 frames):** f0520–f0526,f0550–f0560,f0615.
- **Lost vs iter10 (18 frames):** f0561–f0577,f0618.

SPEC frame checks:
- **PASS** f520–526 (recovered R, expected present): present f0520–f0526; absent —.
- **PASS** f557–560 (recovered R, expected present): present f0557–f0560; absent —.
- **PASS** f561–575 (iter10 zigzag, expected absent): absent f0561–f0575; unexpectedly present —.
- **PASS** f165–166 (still missing): absent f0165–f0166; unexpectedly present —.

Of the 123 iter11 trimmed misses at z = 10: **71** still have raw `main_R` at z = 10 (trimmed away), **8** have raw starting beyond z 10, **44** have no raw `main_R`.

## Frame 0: lanes and road edges present

Presence at z = 10 using the same `x_at` / `side_series` definition as the L10/R10 metrics, on frame 0 of every clip.

| clip | recorded frame | lane L10 | lane R10 | edge L10 | edge R10 | n_lanes | n_edges |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sample | f0000 | yes | yes | yes | yes | 2 | 2 |
| exit_approach | f0000 | yes | yes | yes | yes | 2 | 2 |
| late | f0000 | yes | yes | yes | yes | 2 | 2 |
| final | f0000 | yes | yes | yes | yes | 2 | 2 |
| city | f0000 | yes | yes | yes | yes | 4 | 2 |
| lane_split | f0000 | yes | NO | yes | yes | 1 | 2 |
| munich | f0000 | yes | yes | yes | NO | 2 | 2 |
| a2d2a | f0000 | NO | yes | yes | NO | 1 | 1 |
| a2d2b | f0000 | NO | NO | NO | yes | 0 | 2 |

On the six KITTI highway and city clips (sample, exit, late, final, city) plus exit's partner set, **sample / exit_approach / late / final / city all have both lanes and both road edges at z = 10 on frame 0** (iter10 had lanes but no edges on those frames; iter11 backfilled the edges).

The other clips are not complete at z = 10 on frame 0 (this is reported, not a presence-drop flag):
- **lane_split:** left lane and both road edges are present (edges were backfilled). The right lane is still absent at f0, as in iter10 (only one lane record; clip-level R presence stays 73.3%).
- **munich:** both lanes and the left edge are present at z = 10. `main_R` was backfilled but starts at z = 17.0, so it is not present at z = 10.
- **a2d2a:** right lane and left edge present; no left lane (same as iter10) and no right edge at z = 10.
- **a2d2b:** no lanes (same as iter10). Right edge present at z = 10; left edge was backfilled but starts at z = 11.5.

## Lane presence and width jitter vs YOLO (KITTI clips)

YOLOPv2 is the measurement baseline only (no road edges). Numbers are L10/R10 presence (%), lane wjit, and lane jit2 L/R at z = 10, from `metrics_yolo_vs_run.json` (YOLO) and this run's `metrics.json` (iter10 → iter11).

| clip | lane pres L/R YOLO · iter10 → iter11 | lane wjit YOLO · iter10 → iter11 | lane jit2 L YOLO · iter10 → iter11 | lane jit2 R YOLO · iter10 → iter11 |
| --- | --- | --- | --- | --- |
| sample | 100.0/100.0 · 100.0/100.0 → 100.0/100.0 | 0.006 · 0.004 → 0.001 | 0.019 · 0.004 → 0.002 | 0.005 · 0.003 → 0.002 |
| exit | 100.0/100.0 · 100.0/100.0 → 100.0/100.0 | 0.007 · 0.017 → 0.007 | 0.013 · 0.005 → 0.005 | 0.009 · 0.014 → 0.008 |
| late | 100.0/100.0 · 97.5/100.0 → 97.5/100.0 | 0.009 · 0.019 → 0.006 | 0.014 · 0.011 → 0.011 | 0.014 · 0.007 → 0.012 |
| final | 100.0/100.0 · 93.3/90.0 → 100.0/100.0 | 0.008 · 0.012 → 0.006 | 0.024 · 0.016 → 0.014 | 0.017 · 0.004 → 0.004 |
| city | 100.0/100.0 · 100.0/100.0 → 100.0/100.0 | 0.005 · 0.004 → 0.001 | 0.001 · 0.002 → 0.002 | 0.004 · 0.003 → 0.001 |
| lane_split | 76.7/26.7 · 100.0/73.3 → 100.0/73.3 | 0.092 · 0.131 → 0.131 | 0.257 · 0.011 → 0.011 | 0.721 · 0.022 → 0.022 |

Versus YOLO on this GPU run:
- **sample:** pres L/R 100.0/100.0 (YOLO 100.0/100.0); wjit 0.001 (YOLO 0.006); jit2 L/R 0.002/0.002 (YOLO 0.019/0.005).
- **exit:** pres L/R 100.0/100.0 (YOLO 100.0/100.0); wjit 0.007 (YOLO 0.007); jit2 L/R 0.005/0.008 (YOLO 0.013/0.009).
- **late:** pres L/R 97.5/100.0 (YOLO 100.0/100.0); wjit 0.006 (YOLO 0.009); jit2 L/R 0.011/0.012 (YOLO 0.014/0.014).
- **final:** pres L/R 100.0/100.0 (YOLO 100.0/100.0); wjit 0.006 (YOLO 0.008); jit2 L/R 0.014/0.004 (YOLO 0.024/0.017).
- **city:** pres L/R 100.0/100.0 (YOLO 100.0/100.0); wjit 0.001 (YOLO 0.005); jit2 L/R 0.002/0.001 (YOLO 0.001/0.004).
- **lane_split:** pres L/R 100.0/73.3 (YOLO 76.7/26.7); wjit 0.131 (YOLO 0.092); jit2 L/R 0.011/0.022 (YOLO 0.257/0.721).

