# GPU runner

Commit the finished run onto `main`. Do not rerun the model.

From the machine that has the outputs:

- `data/qwen-a2d2/ret_v6/results/eval.json` → `results/eval.json`
- `data/qwen-a2d2/ret_v6/results/summary.md` → `results/summary.md`

`results/*.npz` stays untracked. Push to `https://github.com/sudeshgm/qwen-drive-a2d2` `main` and reply with the new SHA.
