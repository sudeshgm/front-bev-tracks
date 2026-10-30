# Qwen-Drive on one A2D2 clip

Object tracks and lane tracks from `Qwen/Qwen-Drive-1.0-4B` perception, on a
surround video that is **not** nuScenes.

The released head was trained on a fixed nuScenes 6-camera ring and an
OpenScene/nuPlan 8-camera ring, at 896×512. The authors say a different camera
layout is not covered and will likely degrade. This repo runs that experiment
on Audi A2D2 instead of pretending the rig matches.

## Clip

- Dataset: [A2D2](https://www.a2d2.audi/) sensor-fusion sequence `20180810_150607` (Gaimersheim), CC BY-ND 4.0.
- Window: camera frames `12025` to `12139` step 6 (20 frames, 5 Hz, about 3.8 s).
- Motion: about 66 km/h on a marked two-lane road, solid edge line and dashed lane line. The earlier Munich window (`20190401_121727`, frame 8472) is cobblestone and is no longer the default.
- Six cameras, undistorted to the pinhole `CamMatrix`, then labelled with the nuScenes view tags in this order: front center, front right, side right, rear center, side left, front left.

The side cameras look near ±90°, and the two "front corner" cameras look almost straight ahead. Those slots are the closest nuScenes names, not a matched rig. Lidar frame is set equal to the vehicle frame (x forward, y left, z up) so Qwen's ego map grid lines up with the bus.

## GPU runner

Kaggle T4, fp16, PyTorch SDPA. No FlashAttention. The perception CUDA kernel is bf16-only, so the job patches it to the upstream PyTorch fallback and loads `perception/` as a subfolder of `Qwen/Qwen-Drive-1.0-4B` (not as its own repo id).

On T4 x2 the VLM stays on `cuda:0` and the perception head runs on `cuda:1` (`--perception-device`, default `auto`); one T4 cannot hold both through voxel pooling. The job sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The BEV path runs with `cuda:1` as the current device, and `voxel_pool_depth` is wrapped with a device guard and index checks. Deformable attention on fp16 goes through `qdtrack/msda.py` (fp32 grid math clamped to [-2, 2], fp16 sampling, fp32 accumulation, query-chunked), calibration is handed to the head as fp32 tensors, and boxes are decoded in fp32. `QD_PERCEPTION_DEVICE=cuda:0` forces one GPU; `QD_CUDA_DEBUG=1` sets `CUDA_LAUNCH_BLOCKING=1` and logs each voxel-pool call.

Run `main`. Do not check out `243b9a2` — that commit is the one that failed on the hub id.

```bash
git clone https://github.com/sudeshgm/qwen-drive-a2d2
cd qwen-drive-a2d2
bash scripts/gpu_job.sh
```

The job downloads the weights and the Audi frames, writes `results/<token>.npz`, and `results/eval.json`. Load that JSON in the evaluation UI.

`eval.json` has:

- `frames[].detections` — per-frame boxes `[x, y, z, w, l, h, yaw, vx, vy]` in the ego frame, score floor 0.2
- `object_tracks` — every detection class, kept across frames (nearest-first constant-velocity, class-gated, miss gap of 3 frames)
- `lane_tracks` — `road_line` and `road_edge` polylines associated in the odometry frame (chamfer gate)

## Later: domain-shift stack

Not this run. After Qwen writes `results/eval.json`, the same clip and the same `eval.json` schema get a second pass that does not use a nuScenes bird's-eye-view grid.

Undistort to the pinhole, then per camera:

- depth: UniDepth or Metric3D v2, using these intrinsics
- objects: YOLO-World
- lanes: YOLOPv2

Lift pixels into the ego frame with the extrinsics and the metric depth. Reuse the existing object and lane tracker. SparseDrive and MapTR stay off the list. They freeze the nuScenes rig.

## CPU checks

```bash
python scripts/test_tracks.py
```

Needs the A2D2 calibration JSON at `/tmp/a2d2/cams_lidars.json` for the projection test (or skip that one). No GPU.
