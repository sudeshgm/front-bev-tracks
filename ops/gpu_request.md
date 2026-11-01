# GPU runner

Run the depth-agreement pass again. Do not run `scripts/gpu_job.sh`. Do not run `scripts/gpu_job_lift.sh`. Do not rerun Qwen-Drive.

The last run, `d37a814`, kept 1% of the lane points and no boxes. UniDepth's metres were off by more than 30% almost everywhere, so the check was measuring a scale bias. Current `main` fits one scale per frame on the lane pixels between 8 m and 22 m. Inside 22 m the point comes from the ground plane. Past 22 m it comes from the scaled depth, and only if that is within 40% of the plane.

```bash
bash scripts/gpu_job_fuse.sh
```

Same Gaimersheim window already packed: sequence `20180810_150607`, frames `12025` through `12139`, 115 frames. The script fetches that pack only if `data/frames` is missing. It does not install torch. It does not load YOLO-World.

When it exits 0, upload `results/eval.json` and `results/summary.md` to the Drive folder `qwen-a2d2-fuse`. Reply with the SHA, the frame count, the track counts, the median depth scale, and the kept-point lines. If it fails, send the traceback and stop. Do not start a second run until `main` changes.
