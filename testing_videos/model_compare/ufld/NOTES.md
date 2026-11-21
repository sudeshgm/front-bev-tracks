# UFLD: repo script not runnable as committed (ffcd055)

`scripts/model_compare/ufld_overlay.py` is a skeleton:
1. On Kaggle it first fails with `ModuleNotFoundError: No module named 'addict'` (UFLD's utils/config.py needs `addict`; not installed).
2. Even with addict, `from utils.common import merge_config, get_model` fails: UFLD v1 (cfzd/Ultra-Fast-Lane-Detection @353df10) has no `get_model`, and its `merge_config()` takes no arguments (it parses sys.argv).
3. The frame loop writes the input frame unchanged ("identity until real inference is wired"), so even a successful run would produce no lane output.

Missing pieces: build `model.model.parsingNet(pretrained=False, backbone='18', cls_dim=(201,18,4))`, load `culane_18.pth['model']`, resize to 800x288 with ImageNet normalisation, and decode with `data.constant.culane_row_anchor` as in the repo's demo.py.
Weights culane_18.pth (Google Drive id 1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq), sha256 7cb18e128fda9f7f6be36a338dd8e7a2a948ce2a03dad878f0e651409b4a5b51, are staged in private Kaggle dataset hiddenslave/kitti-fc-compare-inputs.
