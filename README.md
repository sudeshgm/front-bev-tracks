# Front and bird's-eye tracks

A front-camera driving video goes in. A video comes out: the same view with lane lines and tracked vehicles, and a bird's-eye panel beside it.

The clip is not stored here. Each run needs a video, a motion file, and a camera file.

```bash
git clone https://github.com/sudeshgm/front-bev-tracks
cd front-bev-tracks
python scripts/overlay.py \
    --video clip.mp4 \
    --motion clip_motion.csv \
    --camera clip_camera.json \
    --weights yolopv2.pt \
    --output overlay.mp4
```

One output frame per motion row. CPU is enough. Needs PyTorch, OpenCV, and ffmpeg.

`--weights` is a TorchScript YOLOPv2. Input is `1x3x640x640` RGB in 0–1. It must return `(detections, drive, lanes)`: three detection scales with 85 channels, and a lane mask on that same 640 canvas. The image is letterboxed into rows 140:500. Pass `--cache` to reuse detections, and `--recompute` to ignore the cache.

`--repeat 2` when the file stores each camera frame twice. The motion file still has one row per camera frame.

Pass `--frames` instead of `--video` when you already have one colour image per motion row (PNG or JPEG, sorted by filename). OpenCV reads them as BGR. That avoids encoding the frames to a video and decoding them again before the detector. The output video is tagged BT.709, limited range. Do not boost saturation before detection; the detector must see the camera's own colour.

## Motion file

CSV with header `t_s,speed_mps,yaw_rate_rps`. One row per camera frame. `t_s` is seconds and must increase. `speed_mps` is the ego speed. `yaw_rate_rps` is radians per second, positive to the left. Axes are ISO 8855: x forward, y left, z up.

## Camera file

JSON. Intrinsics match the video's image size. `position_m` is the camera in the vehicle frame, in metres. The image is a CylinderV01 virtual view, already unrotated, so the view rotation is identity and there is no extra lens term.

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

A ground point in the camera frame projects as `u = cx - fx·atan2(y, x)` and `v = cy - fy·z / hypot(x, y)`. There is no pixel when `x <= 0`. The lift is the inverse of that onto the ground plane `z = 0`.

Vehicles are tracked with a constant-velocity Kalman, forward and back. A vehicle whose sideways motion only follows the viewing angle, and which has not moved along the road, is held at the position from the most head-on views. That keeps a parked vehicle from sliding as the ego vehicle passes it. Lane lines are associated in the same odometry frame.

## Check

```bash
python scripts/test_tracks.py
```

No GPU. One projection check reads a local calibration file if that file is present.
