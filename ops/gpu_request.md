# GPU runner

Run the lift pass again. Do not run `scripts/gpu_job.sh`. Do not rerun Qwen-Drive.

The last run died on a negative-stride copy in the lane step. That line now copies the array first. Run current `main`.

```bash
bash scripts/gpu_job_lift.sh
```

Same Gaimersheim window already packed: sequence `20180810_150607`, frames `12025` through `12139`, 115 frames. The script fetches that pack only if `data/frames` is missing. It does not install torch.

The job is YOLO-World for boxes, YOLOPv2 for lanes, UniDepth V2 for metric depth, lifted with this clip's calibration. When it exits 0, upload `results/eval.json` and `results/summary.md` to the Drive folder `qwen-a2d2-lift`. Reply with the SHA, the frame count, and the object and lane counts. If it fails, send the traceback. Do not start a second run until `main` changes.
