# GPU runner

Run the depth-agreement pass. Do not run `scripts/gpu_job.sh`. Do not run `scripts/gpu_job_lift.sh`. Do not rerun Qwen-Drive.

The 2D lanes and boxes from YOLOPv2 are the ones that looked right. This job keeps a 3D point only when UniDepth's range and the ground-plane range from this camera's calibration agree within 30%. Front camera only. It then tracks those points.

```bash
bash scripts/gpu_job_fuse.sh
```

Same Gaimersheim window already packed: sequence `20180810_150607`, frames `12025` through `12139`, 115 frames. The script fetches that pack only if `data/frames` is missing. It does not install torch. It does not load YOLO-World.

When it exits 0, upload `results/eval.json` and `results/summary.md` to the Drive folder `qwen-a2d2-fuse`. Reply with the SHA, the frame count, the object and lane counts, and the kept-point lines from `summary.md`. If it fails, send the traceback and stop. Do not start a second run until `main` changes.
