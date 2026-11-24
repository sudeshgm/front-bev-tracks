# iter5b: NumPy-1.x hotfix + stage-2-only rerun of the 5 failed iter5 clips (FINAL)

The iter5 run had rc=3. Stage 2 crashed in `lines_mt.fork_gores` because `np.trapezoid` does not exist on the Modal image's NumPy 1.x.
iter5b is iter5 plus a compat fix and a dets-reuse switch. It has **no algorithm changes**; those wait for iter6. patch.diff is against iter5_spec.

## Changes
- `src/lines_mt.py`: `_trapz = getattr(np, "trapezoid", None) or np.trapz`, used at both fork-gore area sites.
  A grep of src/ finds no other NumPy-2-only calls, and the test below guards against new ones.
- `src/run_iter5b.py` (the iter5 kernel, renamed `iter5` → `iter5b`):
  - `FBEV_REUSE_DETS=<dir>` skips the WildDet3D install, pre-flight and stage 1, and copies `wd3d_<tag>.npz`
    (or `<tag>_dets_ego.json` as a fallback) into `$FBEV_OUT/dets/`.
  - If any requested clip is missing from that directory, the kernel exits with rc=28 (`reuse_dets_missing`).
  - The SAM2.1/DAV2 install still runs. scipy is pip-installed only if missing (stage 1 used to install it).
  - `FBEV_TAGS=<comma list>` restricts which clips run.
- `cams_iter5b.json` is a copy of cams_iter5.json.

## GPU Bot run (stage 2 only)
- Entry: `src/run_iter5b.py`. Dataset = this folder: `src/`, `cams_iter5b.json`, `yolo_baseline/`, plus the same clip mp4s and cams/motion files as iter5.
- Env vars:
  - `FBEV_REUSE_DETS=<gpubot-bev-iter5 volume>/dets`: the iter5 stage-1 `wd3d_<tag>.npz` files, i.e. iter5's `$FBEV_OUT/dets`.
  - `FBEV_TAGS=kitti_city_cyclist_candidate,kitti_lane_split_candidate,munich,a2d2a,a2d2b`
  - `FBEV_EXTRA_STILLS=6,7,15,18,34,63,86,133,180,450,697` (the iter4/iter5 still set; the automatic picks are kept too).
  - `FBEV_OUT`, `FBEV_TMP`, `FBEV_DS` as in iter5.
  - Leave `FBEV_YOLOPV2` / `FBEV_YOLO_REPO` **unset**: the iter5 YOLO baseline already completed (rc 0), so reuse `iter5/yolo_baseline`.
  - Do not set `FBEV_SMOKE` or `FBEV_MAXF`.
- Outputs:
  - `results/<clip>_foundation_bev_iter5b.mp4` for the 5 clips.
  - `stills/`:
    - kitti_lane_split_candidate f0006/f0015/f0018
    - kitti_city_cyclist_candidate f0006/f0007/f0015
    - munich f0180/f0450/f0697
    - a2d2a f0034/f0086/f0133
    - a2d2b f0034/f0063/f0086
  - Also `tracks/`, `polylines/`, `dets_dump/`, `metrics.md` and `run_meta.json`.
- Expected time: about **25 GPU-min** on a T4.
  - Stage 2 is ~17–19 min: in iter5 these clips took 33 + 35 + 716 + 109 + 131 s before they crashed, and iter4 needed 1009 s for the same clips.
  - Image/SAM2/DAV2 install is ~5 min, and metrics ~1 min.
  - Without dets reuse (a stage 1+2 run of these 5 clips): about 80 min.
- **Merge:** after the run, build one **9-clip `metrics_vs_iter4.md`** from 4 iter5 clips + 5 iter5b clips against iter4:
  - iter5: kitti_highway_sample, exit_approach, late, final;
  - iter5b: city, lane_split, munich, a2d2a, a2d2b.
  - Command: `smoothness_metrics.py --run <iter4> --label iter4 --run <merged dir with the 9 tracks/polylines> --label iter5`.
  - Also merge `ego_speed_table.md`, and keep `metrics_yolo_vs_run.md` against the merged 9 clips.
  - Keep volume gpubot-bev-iter5 until the merged run is verified.

## Tests (CPU, run 24 Nov ~03:45 IST)
- `tests/test_np1_compat.py`:
  - builds a fork-gore polygon and checks its area against the shoelace area, under NumPy-1 (no `np.trapezoid`) and NumPy-2 (no `np.trapz`) APIs via monkeypatch;
  - greps src for NumPy-2-only calls.
- pytest: 11 passed with plain NumPy 2.5.5, and 11 passed under `PYTHONPATH=tests/np1_sim`.
  The `np1_sim/sitecustomize.py` deletes `np.trapezoid`, provides `np.trapz`, and drops `trapezoid` from `__all__`.
  The 4 legacy test scripts also print ALL PASSED under the simulation.
- `cpu_smoke/run_cpu_smoke.sh` runs lane_split under the NumPy-1 simulation (iter4 dets_dump, classical masks, full 30 frames).
  - iter5b: rc 0, and fork gores are built on f24–29 (so the trapz path runs).
  - The same command on iter5_spec reproduces the GPU crash (`AttributeError ... 'trapezoid'`).
- Kernel smoke under the NumPy-1 simulation (`FBEV_SMOKE=1 FBEV_MAXF=8 FBEV_REUSE_DETS=iter4/dets_dump FBEV_TAGS=<5 clips>`): rc 0, 5 × `*_foundation_bev_iter5b.mp4`.
  - The missing-dets path gives rc 28.
  - Note: `smoke_kernel/ds/src` is now a symlink to `src/`. The iter5 local smoke had a stale copy of iter4's src.
