# Metrics versus iter9 (9-clip iter10)

iter9 is the completed GPU run at `/workspace/foundation_bev/iter9/`.
iter10 is this full stage 1+2 GPU run (`/workspace/foundation_bev/iter10/`, pulled from volume `gpubot-bev-iter10`). Both scored KITTI and A2D2 at capture_hz (10 fps) and Munich at 30 fps, taken from each clip's polylines JSON and `metrics.json`.
Gore frames are counts of frames with a non-empty `gores` list; sources are `stripes` / `fork` / `widen`.
Held-edge counts are the number of frames with at least one edge that has `held: true`.
kerb_occ counts are the number of frames with at least one edge that has `kerb_occ: true`.
Knobs were left at code defaults, including the new iter10 defaults (`FBEV_EDGE_GAP_M=5.0`, `FBEV_EDGE_GAP_SLOPE=0.5`, `FBEV_PAIR_W_S=1.5`, `FBEV_LANE_BACKFILL_S=0.3`, `FBEV_KERB_RAMP_S=0.3`, `FBEV_EDGE_DIVERGE_DEG=45`, `FBEV_KERB_CARS=1`). They were not changed after this comparison.
Caveat (SPEC): stage 1 is not deterministic between runs, so clip accel can move by about ±0.05 independently of this code. iter10 does not touch tracks.

| clip | accel iter9 → iter10 | n_tracks 9→10 | idsw 9→10 | lane L10/R10 pres % | lane L10/R10 jit2 | lane wjit | edge L10/R10 pres % | edge L10/R10 jit2 | edge wjit | gore frames iter9 → iter10 | held-edge frames 9→10 | kerb_occ frames 9→10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample | 0.095 → 0.095 | 1→1 | 0.0→0.0 | 95.0/95.0 → 100.0/100.0 | 0.005/0.003 → 0.004/0.003 | 0.006 → 0.004 | 95.0/95.0 → 95.0/95.0 | 0.020/0.009 → 0.020/0.009 | 0.019 → 0.019 | 0 → 0 | 0→0 | 0→0 |
| exit_approach | 0.546 → 0.546 | 2→2 | 0.0→0.0 | 95.0/95.0 → 100.0/100.0 | 0.005/0.018 → 0.005/0.014 | 0.023 → 0.017 | 95.0/95.0 → 95.0/95.0 | 0.016/0.024 → 0.016/0.024 | 0.048 → 0.048 | 6 (widen 6; f0021–f0026) → 6 (widen 6; f0021–f0026) | 0→0 | 0→0 |
| late | 0.534 → 0.534 | 5→5 | 0.0→0.0 | 92.5/95.0 → 97.5/100.0 | 0.012/0.007 → 0.011/0.007 | 0.020 → 0.019 | 95.0/87.5 → 95.0/87.5 | 0.020/0.015 → 0.020/0.015 | 0.035 → 0.035 | 0 → 0 | 9→9 | 0→0 |
| final | 0.574 → 0.574 | 3→3 | 0.0→0.0 | 93.3/90.0 → 93.3/90.0 | 0.012/0.004 → 0.016/0.004 | 0.013 → 0.012 | 90.0/86.7 → 90.0/86.7 | 0.029/0.025 → 0.029/0.025 | 0.023 → 0.023 | 0 → 0 | 0→0 | 0→0 |
| city | 0.548 → 0.548 | 5→5 | 0.0→0.0 | 93.3/93.3 → 100.0/100.0 | 0.003/0.006 → 0.002/0.003 | 0.007 → 0.004 | 93.3/93.3 → 93.3/93.3 | 0.018/0.010 → 0.021/0.010 | 0.054 → 0.054 | 0 → 0 | 2→2 | 0→0 |
| lane_split | 0.951 → 0.951 | 13→13 | 0.0→0.0 | 93.3/73.3 → 100.0/73.3 | 0.011/0.022 → 0.011/0.022 | 0.131 → 0.131 | 93.3/93.3 → 93.3/93.3 | 0.042/0.042 → 0.042/0.042 | 0.125 → 0.127 | 0 → 0 | 7→7 | 15→15 |
| munich | 0.655 → 0.655 | 87→87 | 0.0→0.0 | 85.7/65.2 → 85.9/65.4 | 0.003/0.002 → 0.003/0.002 | 0.006 → 0.006 | 99.8/84.1 → 99.8/86.2 | 0.010/0.024 → 0.010/0.022 | 0.043 → 0.040 | 6 (fork 6; f0760–f0765) → 11 (fork 6; widen 5; f0760–f0765,f0881–f0885) | 196→189 | 276→341 |
| a2d2a | 0.495 → 0.495 | 23→23 | 0.0→0.0 | 15.1/25.0 → 15.1/26.7 | 0.008/0.007 → 0.008/0.007 | — → — | 95.3/94.8 → 95.9/94.8 | 0.038/0.032 → 0.036/0.030 | 0.070 → 0.065 | 0 → 0 | 34→34 | 60→63 |
| a2d2b | 0.821 → 0.821 | 11→11 | 0.0→0.0 | 8.1/2.3 → 8.1/2.3 | 0.012/0.035 → 0.012/0.035 | — → — | 97.1/97.1 → 97.1/97.1 | 0.038/0.032 → 0.038/0.032 | 0.057 → 0.057 | 0 → 0 | 39→39 | 0→0 |

## Presence drops > 5 points vs iter9, accel rise > 0.05, or idsw > 0 (flag only; knobs unchanged)

A drop is flagged when iter10 lane or edge L10/R10 presence is more than 5 points below iter9, per clip and side. Accel is flagged when iter10 `accel_rms` is more than 0.05 above iter9. idsw is flagged when the proxy is above 0.

**No presence drop > 5 points** versus iter9.

**No accel rise above 0.05** versus iter9. Clip-level `accel_rms` and `n_tracks` are **unchanged** on all 9 clips.

**idsw is 0** on all 9 clips in iter10 and iter9.

Notable non-flag movements (for context, not flags):
- sample lane L10 pres: iter9 95.0 → iter10 100.0 (Δ +5.0).
- sample lane R10 pres: iter9 95.0 → iter10 100.0 (Δ +5.0).
- sample lane wjit: iter9 0.006 → iter10 0.004 (Δ -0.002).
- exit_approach lane L10 pres: iter9 95.0 → iter10 100.0 (Δ +5.0).
- exit_approach lane R10 pres: iter9 95.0 → iter10 100.0 (Δ +5.0).
- exit_approach lane R10 jit2: iter9 0.018 → iter10 0.014 (Δ -0.004).
- exit_approach lane wjit: iter9 0.023 → iter10 0.017 (Δ -0.006).
- late lane L10 pres: iter9 92.5 → iter10 97.5 (Δ +5.0).
- late lane R10 pres: iter9 95.0 → iter10 100.0 (Δ +5.0).
- final lane L10 jit2: iter9 0.012 → iter10 0.016 (Δ +0.004).
- city lane L10 pres: iter9 93.3 → iter10 100.0 (Δ +6.7).
- city lane R10 pres: iter9 93.3 → iter10 100.0 (Δ +6.7).
- city lane R10 jit2: iter9 0.006 → iter10 0.003 (Δ -0.003).
- city lane wjit: iter9 0.007 → iter10 0.004 (Δ -0.003).
- city edge L10 jit2: iter9 0.018 → iter10 0.021 (Δ +0.003).
- lane_split lane L10 pres: iter9 93.3 → iter10 100.0 (Δ +6.7).
- lane_split edge wjit: iter9 0.125 → iter10 0.127 (Δ +0.002).
- munich lane L10 pres: iter9 85.7 → iter10 85.9 (Δ +0.2).
- munich lane R10 pres: iter9 65.2 → iter10 65.4 (Δ +0.2).
- munich edge R10 pres: iter9 84.1 → iter10 86.2 (Δ +2.1).
- munich edge R10 jit2: iter9 0.024 → iter10 0.022 (Δ -0.002).
- munich edge wjit: iter9 0.043 → iter10 0.040 (Δ -0.003).
- a2d2a lane R10 pres: iter9 25.0 → iter10 26.7 (Δ +1.7).
- a2d2a edge L10 pres: iter9 95.3 → iter10 95.9 (Δ +0.6).
- a2d2a edge L10 jit2: iter9 0.038 → iter10 0.036 (Δ -0.002).
- a2d2a edge R10 jit2: iter9 0.032 → iter10 0.030 (Δ -0.002).
- a2d2a edge wjit: iter9 0.070 → iter10 0.065 (Δ -0.005).

## Gore-frame check

SPEC: gores must still fire only on exit_approach f21–26 and munich f760–765.
**FAIL.** The expected gores still fire (exit_approach f0021–f0026 widen; munich f0760–f0765 fork), but munich also grew an unexpected **widen** gore on **f0881–f0885** that was not present on iter9. Those five frames have a full-range main_L/main_R (z 9.5–44.5); main_R is `kerb_occ: true` on all five in iter10 (iter9 had kerb_occ on f0881–f0882 only). No other clip grew a gore.

## Held-edge frames per clip

| clip | iter9 | iter10 | iter10 frame list (ranges) |
| --- | --- | --- | --- |
| sample | 0 | 0 | — |
| exit_approach | 0 | 0 | — |
| late | 9 | 9 | f0014,f0021–f0022,f0033–f0036,f0038–f0039 |
| final | 0 | 0 | — |
| city | 2 | 2 | f0028–f0029 |
| lane_split | 7 | 7 | f0005–f0006,f0018–f0019,f0025,f0028–f0029 |
| munich | 196 | 189 | f0004–f0007,f0009–f0013,f0044,f0054–f0055,f0062–f0067,f0120–f0121,f0136–f0141,f0144–f0147,f0152–f0153,f0156–f0157,f0160–f0166,f0170–f0188,f0202–f0203,f0246–f0247,f0249,f0253–f0254,f0259–f0260,f0264–f0265,f0290–f0293,f0298–f0301,f0336–f0337,f0360–f0361,f0414–f0415,f0422–f0423,f0432–f0433,f0436–f0437,f0456–f0457,f0474–f0475,f0512–f0513,f0515–f0516,f0518–f0519,f0527–f0535,f0544–f0545,f0562–f0567,f0570–f0571,f0573–f0575,f0584–f0587,f0602–f0611,f0616–f0617,f0622–f0623,f0641–f0643,f0648–f0657,f0660–f0665,f0668–f0671,f0688–f0689,f0692–f0693,f0700–f0704,f0706–f0707,f0858–f0859,f0866–f0867,f0870–f0873,f0876–f0877,f0896–f0899 |
| a2d2a | 34 | 34 | f0004–f0006,f0008–f0009,f0011–f0013,f0015–f0017,f0020,f0022,f0024,f0027–f0028,f0059,f0062,f0067–f0068,f0092–f0095,f0108–f0111,f0125,f0130,f0138–f0139,f0144,f0146 |
| a2d2b | 39 | 39 | f0007–f0008,f0010–f0015,f0038–f0040,f0068–f0071,f0082–f0083,f0088–f0089,f0091,f0101–f0104,f0107–f0108,f0126–f0129,f0143–f0151 |

## kerb_occ edge frames per clip

A frame is counted when at least one edge has `kerb_occ: true`. `n_kerb_edges` is the total number of such edge records across frames.

| clip | iter9 | iter10 | n_kerb_edges iter10 | iter10 frame list (ranges) |
| --- | --- | --- | --- | --- |
| sample | 0 | 0 | 0 | — |
| exit_approach | 0 | 0 | 0 | — |
| late | 0 | 0 | 0 | — |
| final | 0 | 0 | 0 | — |
| city | 0 | 0 | 0 | — |
| lane_split | 15 | 15 | 15 | f0015–f0029 |
| munich | 276 | 341 | 341 | f0004–f0082,f0152–f0188,f0252–f0255,f0276–f0298,f0300–f0430,f0448–f0470,f0482–f0484,f0732–f0734,f0762–f0764,f0805,f0830–f0832,f0866–f0890,f0894–f0899 |
| a2d2a | 60 | 63 | 84 | f0015,f0053–f0113,f0155 |
| a2d2b | 0 | 0 | 0 | — |

## Munich right edge at z = 10: trimmed `edges_bev` vs `edges_raw` (vs iter9's 757)

Presence uses the same definition as `edges_R10_pres` in `smoothness_metrics.py`: `x_at(z=10)` on `edges_bev` (nearest right edge) for the trimmed output, and `x_at(z=10)` on `edges_raw.main_R` for the untrimmed main right edge. Iter9 was **757 / 900** present (84.1%; raw 824). SPEC expected a gain of about **+26** (f158–173, f256–267 including f266–267, and f647), to ~87.0%.

- Trimmed `edges_bev` right edge present at z = 10: **776** / 900 (**86.2%**, metrics `edges_R10_pres` 86.2). Missing: **124**.
- Untrimmed `edges_raw.main_R` present at z = 10: **824** / 900 (91.6%). Raw starts beyond z 10: 10 (f0148–f0157). Raw absent: 66.
- Change vs iter9: **757 → 776 (Δ +19)**, short of the expected about +26 (net would have been ~783 / 900 ≈ 87.0%).
- **Gained vs iter9 (30 frames):** f0162–f0164,f0167–f0188,f0256,f0258,f0266–f0267,f0618.
- **Lost vs iter9 (11 frames):** f0520–f0526,f0557–f0560.

SPEC expected-recover list (f0158–f0173, f0256–f0267, f0647; 29 listed frames, “about +26”):
- Already present on iter9 (so not part of a new gain): **13** (f0158–f0161,f0257,f0259–f0265,f0647).
- Newly present on iter10: **14** (f0162–f0164,f0167–f0173,f0256,f0258,f0266–f0267).
- Still missing on iter10: **2** (f0165–f0166).
- Extra gains outside that list: **16** (f0174–f0188,f0618).

Of the 124 iter10 trimmed misses at z = 10: **72** still have raw `main_R` at z = 10 (trimmed away), **8** have raw starting beyond z 10, **44** have no raw `main_R`.

## Lane presence and width jitter vs YOLO (KITTI clips)

YOLOPv2 is the measurement baseline only (no road edges). Numbers are L10/R10 presence (%), lane wjit, and lane jit2 L/R at z = 10, from `metrics_yolo_vs_run.json` (YOLO) and this run's `metrics.json` (iter9 → iter10).

| clip | lane pres L/R YOLO · iter9 → iter10 | lane wjit YOLO · iter9 → iter10 | lane jit2 L YOLO · iter9 → iter10 | lane jit2 R YOLO · iter9 → iter10 |
| --- | --- | --- | --- | --- |
| sample | 100.0/100.0 · 95.0/95.0 → 100.0/100.0 | 0.006 · 0.006 → 0.004 | 0.019 · 0.005 → 0.004 | 0.005 · 0.003 → 0.003 |
| exit | 100.0/100.0 · 95.0/95.0 → 100.0/100.0 | 0.007 · 0.023 → 0.017 | 0.013 · 0.005 → 0.005 | 0.009 · 0.018 → 0.014 |
| late | 100.0/100.0 · 92.5/95.0 → 97.5/100.0 | 0.009 · 0.020 → 0.019 | 0.014 · 0.012 → 0.011 | 0.014 · 0.007 → 0.007 |
| final | 100.0/100.0 · 93.3/90.0 → 93.3/90.0 | 0.008 · 0.013 → 0.012 | 0.024 · 0.012 → 0.016 | 0.017 · 0.004 → 0.004 |
| city | 100.0/100.0 · 93.3/93.3 → 100.0/100.0 | 0.005 · 0.007 → 0.004 | 0.001 · 0.003 → 0.002 | 0.004 · 0.006 → 0.003 |
| lane_split | 76.7/26.7 · 93.3/73.3 → 100.0/73.3 | 0.092 · 0.131 → 0.131 | 0.257 · 0.011 → 0.011 | 0.721 · 0.022 → 0.022 |

Versus YOLO on this GPU run:
- **Presence:** sample, exit and city reach **100/100**, matching YOLO. late is **97.5/100** (YOLO 100/100). lane_split L reaches **100** (ahead of YOLO's 76.7/26.7 on L/R); lane_split R stays 73.3. final stays **93.3/90** (YOLO 100/100; SPEC: its right lane is born at f5, after the 0.3 s backfill window).
- **Width jitter (wjit):** sample 0.004 vs YOLO 0.006; exit 0.017 vs YOLO 0.007; late 0.019 vs YOLO 0.009; final 0.012 vs YOLO 0.008; city 0.004 vs YOLO 0.005; lane_split 0.131 vs YOLO 0.092. Exit, late, final and lane_split remain above YOLO; city is at or below YOLO.
- Left-lane jit2 stays at or below YOLO on every KITTI clip. Exit right-lane jit2 is 0.014 vs YOLO 0.009.
