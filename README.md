# Qwen-Drive on one A2D2 clip

Object tracks and lane tracks from `Qwen/Qwen-Drive-1.0-4B` perception, on a
surround video that is **not** nuScenes.

The released head was trained on a fixed nuScenes 6-camera ring and an
OpenScene/nuPlan 8-camera ring, at 896×512. The authors say a different camera
layout is not covered and will likely degrade. This repo runs that experiment
on Audi A2D2 instead of pretending the rig matches.

## Clip

- Dataset: [A2D2](https://www.a2d2.audi/) sensor-fusion sequence `20180810_150607` (Gaimersheim), CC BY-ND 4.0.
- Window: camera frames `12025` through `12139`, every frame (30 Hz, 115 frames, about 3.8 s). The first pass used every 6th frame only so a free T4 would finish in a few minutes.
- Motion: about 66 km/h on a marked two-lane road, solid edge line and dashed lane line. The earlier Munich window (`20190401_121727`, frame 8472) is cobblestone and is no longer the default.
- Six cameras, undistorted to the pinhole `CamMatrix`, then labelled with the nuScenes view tags in this order: front center, front right, side right, rear center, side left, front left.

The side cameras look near ±90°, and the two "front corner" cameras look almost straight ahead. Those slots are the closest nuScenes names, not a matched rig. Lidar frame is set equal to the vehicle frame (x forward, y left, z up) so Qwen's ego map grid lines up with the bus.

## GPU bot

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

## Front and bird's-eye overlay

This does not use Qwen-Drive. It runs on any front-camera clip once three files are next to it. Nothing about a particular clip is in the repo.

```bash
python scripts/overlay.py \
    --video clip.mp4 \
    --motion clip_motion.csv \
    --camera clip_camera.json \
    --weights yolopv2.pt \
    --output overlay.mp4
```

The output is the front image with lane lines and tracked boxes, and a bird's-eye panel beside it. One output frame per motion row. CPU is enough. Needs PyTorch, OpenCV, and ffmpeg.

`--weights` is a TorchScript YOLOPv2. Input is `1x3x640x640` RGB in 0–1. It must return `(detections, drive, lanes)`: three YOLO scales with 85 channels, and a lane mask on that same 640 canvas. The image is letterboxed into rows 140:500. Pass `--cache` to reuse detections, and `--recompute` to ignore the cache.

`--repeat 2` when the file stores each camera frame twice. The motion CSV still has one row per camera frame.

### Motion CSV

Header `t_s,speed_mps,yaw_rate_rps`. One row per camera frame. `t_s` is seconds and must increase. `speed_mps` is the ego speed. `yaw_rate_rps` is radians per second, positive to the left (ISO 8855: x forward, y left, z up).

### Camera JSON

Intrinsics are for the video's image size. `position_m` is the camera in the vehicle frame, metres, x forward, y left, z up. The view is the CylinderV01 virtual image, already unrotated, so there is no extra distortion and the view rotation is identity.

```json
{
  "fx": 0.0,
  "fy": 0.0,
  "cx": 0.0,
  "cy": 0.0,
  "position_m": [0.0, 0.0, 0.0],
  "width": 1280,
  "height": 720
}
```

A ground point in the camera frame projects as `u = cx - fx·atan2(y, x)` and `v = cy - fy·z / hypot(x, y)`. No pixel when `x <= 0`. The lift is the inverse of that onto the ground plane `z = 0`.

Tracked cars use a constant-velocity Kalman, forward and back. A car whose sideways motion only follows the viewing angle, and which has not moved along the road, is held at the position from the most head-on views. That is what stops a parked car from sliding as the ego car passes it.

