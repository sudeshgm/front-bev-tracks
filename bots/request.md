# GPU bot

Run Anchor3DLane again. Do not run `scripts/gpu_job.sh`, `scripts/gpu_job_lift.sh`, `scripts/gpu_job_fuse.sh`, or `scripts/gpu_job_wild.sh`. Do not rerun Qwen-Drive, YOLOPv2, UniDepth, or WildDet3D.

The last run died while building mmcv. This script does not install mmcv. A small stub in `compat/mmcv` stands in for it, because Kaggle's Python 3.13 cannot compile mmcv 1.6.

```bash
bash scripts/gpu_job_anchor.sh
```

Same Gaimersheim window: sequence `20180810_150607`, frames `12025` through `12139`, 115 frames, front camera only. The script fetches that pack only if `data/frames` does not already hold 115 frames.

Do not pip-install torch. Do not pip-install mmcv. The script clones `tusen-ai/Anchor3DLane` and downloads `openlane_anchor3dlane.pth` from Hugging Face.

When it exits 0, upload `results/eval.json` and `results/summary.md` to the Drive folder `qwen-a2d2-anchor3d`. Reply with the SHA, the frame count, the lane count, and the two gap lines from `summary.md`. If it fails, send the traceback and stop. An out-of-memory stop is a result, not a reason to retry. Do not start a second run until `main` changes.
