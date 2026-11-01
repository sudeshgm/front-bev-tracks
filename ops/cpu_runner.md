# CPU side

Owns https://github.com/sudeshgm/qwen-drive-a2d2. Does not run the model.
The GPU runner owns the Kaggle notebook https://www.kaggle.com/code/kaggle-user/qwen-a2d2 (private, T4).

## Loop

1. Read `main`. The job is `bash scripts/gpu_job_fuse.sh`. Do not pin an old commit. Do not run `scripts/gpu_job.sh` or `scripts/gpu_job_lift.sh`.
2. Hand the GPU runner the current SHA and the bash cell below. Tell it to start the notebook, then check the log within 3 minutes and again every 5 minutes until `results/eval.json` exists or a new traceback appears.
3. The GPU runner must reply with the SHA it ran, `rc`, and either the first error or the counts from `results/summary.md`. It must not wait for the project lead.
4. On a traceback, fix `main`, push, and hand the new SHA back. Stop after `eval.json` or four failed rounds, and tell the project lead which.

## Cell

```bash
cd /kaggle/working
rm -rf qwen-drive-a2d2
git clone https://github.com/sudeshgm/qwen-drive-a2d2
cd qwen-drive-a2d2
git log -1 --format=%H > /kaggle/working/commit.txt
nvidia-smi > /kaggle/working/nvidia_smi.txt
start=$(date +%s)
bash scripts/gpu_job_fuse.sh 2>&1 | tee /kaggle/working/gpu_job.log
echo "rc=${PIPESTATUS[0]} wall_s=$(( $(date +%s)-start ))" > /kaggle/working/run_meta.txt
cp -r results /kaggle/working/results || true
cd /kaggle/working && tar czf qwen_a2d2_results.tgz results gpu_job.log run_meta.txt nvidia_smi.txt commit.txt
```

Constraints: fp16 is not required for this job, SDPA, no FlashAttention, do not pip-install torch. `transformers` must stay `5.14.1`. Keep `commit.txt` on one line.

## Open request

`ops/gpu_request.md` tells the GPU runner to run `bash scripts/gpu_job_fuse.sh` from current `main`. The Qwen run and the earlier lift run are finished.
