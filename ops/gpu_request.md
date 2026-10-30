# GPU runner

Run the Gaimersheim road again from current `main`, at the camera rate. Do not reuse the 20-frame pack.

`bash scripts/gpu_job.sh` now fetches sequence `20180810_150607`, frames `12025` through `12139`, stride 1 (115 frames, 30 Hz). The previous run was every 6th frame.

This is about six times the last job, on the order of 40 minutes on the free T4 x2. When it exits 0, upload `results/eval.json` and `results/summary.md` to the Drive folder `qwen-a2d2-gaimersheim-30hz`. Reply with the SHA, the frame count, and the speed line.
