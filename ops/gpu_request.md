# GPU runner

Run WildDet3D. Do not run `scripts/gpu_job.sh`. Do not run `scripts/gpu_job_lift.sh`. Do not run `scripts/gpu_job_fuse.sh`. Do not rerun Qwen-Drive, YOLOPv2, or UniDepth.

This is a different model. It predicts a 3D centre for each car. The ground-plane lift is not used.

```bash
bash scripts/gpu_job_wild.sh
```

Same Gaimersheim window: sequence `20180810_150607`, frames `12025` through `12139`, 115 frames, front camera only. The script fetches that pack only if `data/frames` is missing.

Do not pip-install torch. The script installs vis4d and compiles `vis4d_cuda_ops`, clones `allenai/WildDet3D`, and downloads one 4.7 GB checkpoint. That install is slow. Check the log, and do not start a second run if the first is still installing.

When it exits 0, upload `results/eval.json` and `results/summary.md` to the Drive folder `qwen-a2d2-wilddet`. Reply with the SHA, the frame count, the box count, and the median range. If it fails, send the traceback and stop. An out-of-memory stop is a result, not a reason to retry. Do not start a second run until `main` changes.
