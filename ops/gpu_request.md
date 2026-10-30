# GPU runner

Run the Gaimersheim road clip from current `main`. Do not reuse the Munich cobblestone pack.

The job is `bash scripts/gpu_job.sh`. It deletes `data/frames` unless `data/frames/f000012025/images` is already there, then downloads sequence `20180810_150607` frames `12025` step 6.

When it exits 0, upload `results/eval.json` and `results/summary.md` to the Drive folder `qwen-a2d2-gaimersheim`, or commit them on `main`. Reply with the SHA and the speed line from the log.
