# TwinLiteNet on the six KITTI front-camera clips (raw output, no tracking)

- Script: `scripts/model_compare/twinlite_overlay.py` from sudeshgm/front-bev-tracks, branch lane-solver-fast, commit ffcd0558048e7ef3f02b531629afdc57645496d6. Run unmodified.
- Model: TwinLiteNet (chequanghuy/TwinLiteNet, commit 524141821c373c886443cf4254e3e87787c2c55e), **MIT licence**. Weights `pretrained/best.pth` from that repo, sha256 98ffc6c8abc98fda72052117249ce4e2dbc85e58cfcfc6ee2d42f5398c772463 (BDD100K-trained: drivable area + lane lines).
- Overlay: green = drivable area, red = lane lines, blended 60/40 by the script. Input is resized to 640x360 by the script, then masks are resized back to 640x480.
- Inputs: the six clips from Drive folder KITTI_BEV_Test_Videos (640x480, 4 fps, 30-40 frames each). Output is 640x480, 4 fps, same frame count, mpeg4 (mp4v) codec as the script writes it; some browsers will not play mp4v, VLC will.
- Compute: free Kaggle T4 (hiddenslave, private kernel `kitti-fc-twinlite-repo` v1), fp32, PyTorch 2.11.0+cu130, OpenCV 4.14.0. run_meta rc=0, whole kernel wall 80 s (includes git clones and a failed UFLD attempt). Per-clip wall incl. model load: see timing.json (4-11 s).
- Speed: the script does not time inference. A separate timing run of the same network/weights on the same T4 gave about 12-14 ms per frame (about 70-80 FPS, fp32, batch 1).
- Checksums: SHA256SUMS_out.txt (verified on the box). $0 paid.
