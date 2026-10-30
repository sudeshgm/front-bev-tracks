# GPU runner

Put the finished run where the evaluation side can read it. Do not rerun the model.

Either is enough:

1. Commit onto `main` of `https://github.com/sudeshgm/qwen-drive-a2d2`:
   - `data/qwen-a2d2/ret_v6/results/eval.json` → `results/eval.json`
   - `data/qwen-a2d2/ret_v6/results/summary.md` → `results/summary.md`
   - Leave `results/*.npz` untracked. Reply with the SHA.
2. Or upload those two files to the project lead's Google Drive, in a folder named `qwen-a2d2-ret_v6`. Reply with both file links.

Do not send only a path on the GPU machine. That disk is not visible from here.
