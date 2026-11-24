# Metrics versus iter8 (9-clip iter9)

iter8 is the completed GPU run at `/workspace/foundation_bev/iter8/`.
iter9 is this full stage 1+2 GPU run (`/workspace/foundation_bev/iter9/`, pulled from volume `gpubot-bev-iter9`). Both scored KITTI and A2D2 at capture_hz (10 fps) and Munich at 30 fps, taken from each clip's polylines JSON and `metrics.json`.
Gore frames are counts of frames with a non-empty `gores` list; sources are `stripes` / `fork` / `widen`.
Held-edge counts are the number of frames with at least one edge that has `held: true`.
kerb_occ counts are the number of frames with at least one edge that has `kerb_occ: true` (iter9 feature; default `FBEV_KERB_CARS=1`).
Per-track accel RMS uses the same estimator as `smoothness_metrics.py` (RMS of 2nd differences of drawn `bev_x`/`bev_z`). Culprit tracks were matched by the SPEC position windows.
Knobs were left at code defaults. `FBEV_RAY_R`, `FBEV_EDGE_DIVERGE_DEG`, and `FBEV_KERB_CARS` were unset (`KERB_CARS` default 1, on). They were not changed after this comparison.
Caveat (SPEC): stage 1 is not deterministic between runs, so clip accel can move by about ±0.05 independently of this code. iter9 does not touch tracks. Use the culprit per-track table to separate the two.

| clip | accel iter8 → iter9 | n_tracks 8→9 | idsw 8→9 | lane L10/R10 pres % | lane L10/R10 jit2 | lane wjit | edge L10/R10 pres % | edge L10/R10 jit2 | edge wjit | gore frames iter8 → iter9 | held-edge frames 8→9 | kerb_occ frames iter9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample | 0.095 → 0.095 | 1→1 | 0.0→0.0 | 95.0/95.0 → 95.0/95.0 | 0.007/0.005 → 0.005/0.003 | 0.007 → 0.006 | 95.0/95.0 → 95.0/95.0 | 0.026/0.018 → 0.020/0.009 | 0.027 → 0.019 | 0 → 0 | 0→0 | 0 |
| exit_approach | 0.546 → 0.546 | 2→2 | 0.0→0.0 | 95.0/95.0 → 95.0/95.0 | 0.007/0.018 → 0.005/0.018 | 0.033 → 0.023 | 95.0/95.0 → 95.0/95.0 | 0.019/0.033 → 0.016/0.024 | 0.055 → 0.048 | 6 (widen 6; f0021–f0026) → 6 (widen 6; f0021–f0026) | 0→0 | 0 |
| late | 0.534 → 0.534 | 5→5 | 0.0→0.0 | 92.5/95.0 → 92.5/95.0 | 0.012/0.019 → 0.012/0.007 | 0.020 → 0.020 | 95.0/87.5 → 95.0/87.5 | 0.024/0.040 → 0.020/0.015 | 0.052 → 0.035 | 0 → 0 | 9→9 | 0 |
| final | 0.574 → 0.574 | 3→3 | 0.0→0.0 | 93.3/90.0 → 93.3/90.0 | 0.039/0.006 → 0.012/0.004 | 0.024 → 0.013 | 90.0/86.7 → 90.0/86.7 | 0.033/0.040 → 0.029/0.025 | 0.033 → 0.023 | 0 → 0 | 0→0 | 0 |
| city | 0.548 → 0.548 | 5→5 | 0.0→0.0 | 93.3/93.3 → 93.3/93.3 | 0.004/0.010 → 0.003/0.006 | 0.008 → 0.007 | 93.3/93.3 → 93.3/93.3 | 0.065/0.026 → 0.018/0.010 | 0.070 → 0.054 | 0 → 0 | 2→2 | 0 |
| lane_split | 0.951 → 0.951 | 13→13 | 0.0→0.0 | 93.3/73.3 → 93.3/73.3 | 0.063/0.025 → 0.011/0.022 | 0.136 → 0.131 | 93.3/93.3 → 93.3/93.3 | 0.069/0.077 → 0.042/0.042 | 0.107 → 0.125 | 0 → 0 | 7→7 | 15 |
| munich | 0.655 → 0.655 | 87→87 | 0.0→0.0 | 85.7/65.3 → 85.7/65.2 | 0.009/0.003 → 0.003/0.002 | 0.010 → 0.006 | 99.8/84.3 → 99.8/84.1 | 0.034/0.042 → 0.010/0.024 | 0.052 → 0.043 | 6 (fork 6; f0760–f0765) → 6 (fork 6; f0760–f0765) | 196→196 | 276 |
| a2d2a | 0.495 → 0.495 | 23→23 | 0.0→0.0 | 16.9/23.3 → 15.1/25.0 | 0.001/0.010 → 0.008/0.007 | — → — | 97.1/94.8 → 95.3/94.8 | 0.044/0.047 → 0.038/0.032 | 0.062 → 0.070 | 0 → 0 | 34→34 | 60 |
| a2d2b | 0.821 → 0.821 | 11→11 | 0.0→0.0 | 8.7/1.7 → 8.1/2.3 | 0.017/0.050 → 0.012/0.035 | — → — | 97.1/97.1 → 97.1/97.1 | 0.059/0.057 → 0.038/0.032 | 0.085 → 0.057 | 0 → 0 | 39→39 | 0 |

## Presence drops > 5 points vs iter8, accel rise > 0.05, or idsw > 0 (flag only; knobs unchanged)

A drop is flagged when iter9 lane or edge L10/R10 presence is more than 5 points below iter8, per clip and side. Accel is flagged when iter9 `accel_rms` is more than 0.05 above iter8. idsw is flagged when the proxy is above 0.

**No presence drop > 5 points** versus iter8.

**No accel rise above 0.05** versus iter8.

**idsw is 0** on all 9 clips in iter9 and iter8.

Clip-level `accel_rms` and `n_tracks` are **unchanged** on all 9 clips (see also the per-track table). Line jitter generally improved, which is the expected effect of iter9's zero-lag line smoothing: for example lane_split L10 jit2 0.063→0.011, final L10 jit2 0.039→0.012, city edge L10/R10 jit2 0.065/0.026→0.018/0.010, munich edge L10/R10 jit2 0.034/0.042→0.010/0.024. lane_split edge wjit moved 0.107→0.125.

Smaller presence movements (not flags):
- a2d2a lane L10 pres: iter8 16.9 → iter9 15.1 (Δ -1.8).
- a2d2a lane R10 pres: iter8 23.3 → iter9 25.0 (Δ +1.7).
- a2d2a edge L10 pres: iter8 97.1 → iter9 95.3 (Δ -1.8).
- a2d2b lane L10 pres: iter8 8.7 → iter9 8.1 (Δ -0.6).
- a2d2b lane R10 pres: iter8 1.7 → iter9 2.3 (Δ +0.6).
- munich lane R10 pres: iter8 65.3 → iter9 65.2 (Δ -0.1).
- munich edge R10 pres: iter8 84.3 → iter9 84.1 (Δ -0.2; 759/900 → 757/900; the two new misses are f0266–f0267).

## Gore-frame check

SPEC: gores must still fire only on exit_approach f21–26 and munich f760–765.
**Pass.** Gore frames fired only on exit_approach f0021–f0026 (widen) and munich f0760–f0765 (fork). No other clip had a gore.

## Held-edge frames per clip

| clip | iter8 | iter9 | iter9 frame list (ranges) |
| --- | --- | --- | --- |
| sample | 0 | 0 | — |
| exit_approach | 0 | 0 | — |
| late | 9 | 9 | f0014,f0021–f0022,f0033–f0036,f0038–f0039 |
| final | 0 | 0 | — |
| city | 2 | 2 | f0028–f0029 |
| lane_split | 7 | 7 | f0005–f0006,f0018–f0019,f0025,f0028–f0029 |
| munich | 196 | 196 | f0004–f0007,f0009–f0013,f0044,f0054–f0055,f0062–f0067,f0120–f0121,f0136–f0141,f0144–f0147,f0152–f0153,f0156–f0157,f0160–f0166,f0170–f0188,f0202–f0203,f0246–f0247,f0249,f0253–f0254,f0259–f0260,f0264–f0265,f0290–f0293,f0298–f0301,f0336–f0337,f0360–f0361,f0414–f0415,f0422–f0423,f0432–f0433,f0436–f0437,f0456–f0457,f0474–f0475,f0512–f0513,f0515–f0516,f0518–f0535,f0544–f0545,f0562–f0567,f0570–f0571,f0573–f0575,f0584–f0587,f0602–f0611,f0616–f0617,f0622–f0623,f0641–f0643,f0648–f0657,f0660–f0665,f0668–f0671,f0688–f0689,f0692–f0693,f0700–f0704,f0706–f0707,f0858–f0859,f0866–f0867,f0870–f0873,f0876–f0877,f0896–f0899 |
| a2d2a | 34 | 34 | f0004–f0006,f0008–f0009,f0011–f0013,f0015–f0017,f0020,f0022,f0024,f0027–f0028,f0059,f0062,f0067–f0068,f0092–f0095,f0108–f0111,f0125,f0130,f0138–f0139,f0144,f0146 |
| a2d2b | 39 | 39 | f0007–f0008,f0010–f0015,f0038–f0040,f0068–f0071,f0082–f0083,f0088–f0089,f0091,f0101–f0104,f0107–f0108,f0126–f0129,f0143–f0151 |

## kerb_occ edge frames per clip (iter9)

A frame is counted when at least one edge has `kerb_occ: true`. `n_kerb_edges` is the total number of such edge records across frames.

| clip | kerb_occ frames | n_kerb_edges | frame list (ranges) |
| --- | --- | --- | --- |
| sample | 0 | 0 | — |
| exit_approach | 0 | 0 | — |
| late | 0 | 0 | — |
| final | 0 | 0 | — |
| city | 0 | 0 | — |
| lane_split | 15 | 15 | f0015–f0029 |
| munich | 276 | 276 | f0004–f0044,f0048–f0066,f0072–f0073,f0078–f0079,f0152–f0188,f0252–f0256,f0276–f0290,f0300–f0314,f0320–f0360,f0364–f0386,f0390–f0391,f0396–f0422,f0448–f0462,f0482–f0483,f0732–f0733,f0762–f0763,f0805,f0830–f0831,f0866–f0882,f0894–f0899 |
| a2d2a | 60 | 78 | f0015,f0053–f0110,f0155 |
| a2d2b | 0 | 0 | — |

## Munich right edge at z = 10: `edges_raw` vs trimmed (diagnosis (c))

SPEC asked how many of the frames where main_R starts beyond z 10 were trimmed, and how many were never there. Counts below use the same presence definition as `edges_R10_pres` in `smoothness_metrics.py`: `x_at(z=10)` on `edges_bev` (nearest right edge) for the trimmed output, and `x_at(z=10)` on `edges_raw.main_R` for the untrimmed main right edge. “Starts beyond z 10” means the raw `main_R` polyline's minimum z is greater than 10 (so it cannot be present at z = 10). Iter8 `edges_R10_pres` was **84.3%** (759/900 present, **141 missing**). Iter9 is **84.1%** (757/900 present, **143 missing**). The two added misses vs iter8 are f0266–f0267; every iter8 miss is still a miss.

- Trimmed `edges_bev` right edge present at z = 10: **757** / 900 (84.1%). Missing: **143**.
- Untrimmed `edges_raw.main_R` present at z = 10: **824** / 900 (91.6%).
- Of the **143** official trimmed misses at z = 10:
  - **Trimmed away** (raw `main_R` *was* present at z = 10, trimmed `edges_bev` is not): **75** frames (f0002–f0012,f0144–f0147,f0162–f0163,f0167–f0173,f0228–f0254,f0256,f0258,f0266–f0267,f0535–f0541,f0550–f0556,f0614–f0618,f0658).
  - **Never there** (raw `main_R` starts beyond z 10): **8** frames (f0148–f0155).
  - **Never there** (no raw `main_R` at all): **60** frames (f0000–f0001,f0013,f0164–f0166,f0174–f0227).
- Raw `main_R` starts beyond z 10 in **10** frames total (f0148–f0157). Two of those (f0156–f0157) still have a trimmed right edge at z = 10 from some other right polyline. Raw `main_R` is absent in 66 frames (the 60 official misses above plus f0606–f0611, which are present in trimmed `edges_bev` via another right edge).

**Diagnosis (c) settled:** the old “75 frames, main_R only starts beyond z 10 (f2–13, f144–162 …)” guess is **not** what the untrimmed geometry shows. Those 75 official misses (including f0002–f0012 and f0144–f0147) have `edges_raw.main_R` already at z = 10 and are then missing from the trimmed `edges_bev`. Only **8** official misses truly start beyond z 10 (f0148–f0155), and **60** have no raw `main_R` at all. So **75 were trimmed, 68 were never there** (8 starts-beyond + 60 raw-absent). The trim/kerb path, not missing pre-trim geometry, accounts for the 75-frame block.

## Culprit per-track accel RMS (iter8 GPU → iter9 GPU)

Estimator: `smoothness_metrics.accel_rms` on drawn `bev_x`/`bev_z` inside the SPEC window (same as the clip-level `accel_rms`). Matched by the SPEC position windows; IDs stayed 31 / 6 / 7 / 8. SPEC: iter9 does not touch tracks. On this GPU run the four culprit tracks are **identical** to iter8 (same IDs, same windowed mean x/z, same accel RMS to machine precision, same sample counts), and every clip-level accel is unchanged.

| clip | want id | matched id 8→9 | frames | mean x/z iter9 | accel RMS iter8 → iter9 | n samples |
| --- | --- | --- | --- | --- | --- | --- |
| a2d2a | 31 | 31→31 | f107–f170 | -12.0/21.4 | 0.43 → 0.43 | 62→62 |
| lane_split | 6 | 6→6 | f0–f29 | 5.0/12.7 | 1.48 → 1.48 | 28→28 |
| lane_split | 7 | 7→7 | f0–f29 | 4.4/14.9 | 1.24 → 1.24 | 28→28 |
| a2d2b | 8 | 8→8 | f63–f100 | -5.1/11.5 | 1.47 → 1.47 | 36→36 |

Position check (iter9 mean in the SPEC window): values are in the table. Matching used the same windows as iter8 (a2d2a 31 around −12/21, f107–170; lane_split 6 and 7 in the 4.1–5.2 / 12–15 band, f0–29; a2d2b 8 near −4.3 to −5.1 / 11–12, f63–100).
