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

Lane lines can come from a second model. `--lane-backend twinlite` runs [TwinLiteNet+](https://github.com/chequanghuy/TwinLiteNetPlus) (MIT) for the lane lines and takes the road edges from the left and right side of its drivable-area mask. Vehicle boxes stay with YOLOPv2. `--lane-code` is a checkout of that repo and `--lane-weights` is its `large.pth`. `--reuse-boxes cache.npz` keeps an existing box cache so the vehicle model is not run again. Road edges are drawn in orange and lane lines in green.

`--wilddet results/wilddet.json` draws a second set of boxes from WildDet3D on top of those tracks. The JSON comes from `scripts/wilddet_clip.py` on a GPU. Each box is labelled with the prompt name. The two numbers after the name are how far the model's 3D centre is from the ground-plane guess, in metres, forward and left. A diamond in the bird's-eye panel is that centre. The ground-plane tracks are unchanged.

## Optional: RF-DETR objects (`--rfdetr`)

Off by default; without the flag every output is byte-identical to before. With `--rfdetr`, [RF-DETR](https://github.com/roboflow/rf-detr) Large (Apache-2.0, `pip install rfdetr==1.12.1`, COCO weights, 704 px, fp32) supplies the objects and YOLOPv2 keeps the lane lines. Each object keeps its own class and colour in both panels: car, truck and bus, and every vulnerable road user COCO has: person, bicycle, motorcycle, and the animals (dog, cat, horse, cow, sheep, bird, elephant, bear, zebra, giraffe). A person on a bicycle box is one `cyclist`, on a motorcycle one `motorcyclist`. Car, truck and bus share the vehicle tracker and each track takes the class most of its detections had, so one frame read as a truck does not break a car track. Person and bicycle/cyclist tracks may also stay linked across a one-frame class flicker, using the same range-tolerant cost as cars; the vehicle association path itself is unchanged. The bird's-eye panel shows a class key.

```bash
python scripts/overlay.py --video clip.mp4 --motion clip_motion.csv --camera clip_camera.json \
    --weights yolopv2.pt --output overlay.mp4 --rfdetr
```

- `--rfdetr-threshold 0.5` score threshold. The box cache keeps every score from 0.2 up, so another threshold needs no rerun.
- `--rfdetr-mode all` (default) takes every object from RF-DETR; `vru` keeps the YOLOPv2 vehicle boxes and adds RF-DETR's vulnerable road users and animals. `label` keeps every YOLOPv2 vehicle box and track exactly as without `--rfdetr` and only gives each vehicle track the class (car, truck, bus, motorcyclist, cyclist, ...) of the RF-DETR box it overlapped most, score-weighted over the track (`kind`, `source: yolopv2`). RF-DETR objects that are not a YOLOPv2 box (VRUs, animals, and cars, trucks or buses only RF-DETR saw) are tracked in a separate pass, with ids from 100000 and `source: rfdetr`; a vehicle among them is marked `rf_only` so vehicle tags can leave it out. `track_label_mode()` returns both sets for callers that do not go through `main()`.
- `--rfdetr-cache` is the RF-DETR box cache (default `<cache>_rfdetr.npz`, reused unless `--recompute`). `--rfdetr-device cuda` runs it on a GPU.

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

Vehicles are tracked with a constant-velocity Kalman, forward and back. The speed comes from the positions. A zero speed written in the box is ignored, because mixing it in stops the prediction and the track breaks at the next gap. Association allows a large error along the road, because the range comes from the bottom of the box and that is noisy when the car is far, and a small error sideways, so a car in the next lane is not the same track. A missed detection can rejoin for about a second. A detection that jumps more than about two metres sideways in one frame is a different car, even when there is no gap: that is the car in the next lane, and the old speed can point straight at it. After the last detection a moving track is continued for up to two seconds, and those points are marked predicted, so a car that leaves the image still reaches the ego path. The predicted points do not take on a new detection. The stored point is the centre, half a length past the face the camera saw, along the track's own heading. The smoother trusts a far range less than a near one, so an approaching car is not left several metres short of where it was detected. The two-second tail keeps the last measured speed and holds that heading relative to the ego, so it does not curve away when the ego turns. A vehicle whose sideways motion only follows the viewing angle, and which has not moved along the road, is held at the position from the most head-on views. A vehicle whose fitted speed is about zero is held still. An oncoming vehicle is not: it is moving, even when the measured range closes at about the ego speed. Cyclists and pedestrians are not in the YOLOPv2 boxes. They come from `--wilddet`, whose prompts include `bicycle` and `person`, and they are drawn in both panels.

Each lane marking is one curve on the ground. The detector supplies the pixels of the marking. A chain joins a curve only when those pixels land on the curve, so a curb or a canopy edge a metre away does not pull the paint. A stretch of the curve with no paint under it is removed, which drops a line drawn across empty road between two clumps. Two curves that cover the same dashes are one marking. The picture shows every marking that comes within about two lanes of the car, so the next lane and a merge stay. A merge is one marking that moves sideways: the two pieces are joined where their ends meet, not by how far apart their middles are. A marking may not pass through the line beside it. Where two curves swap sides, the tails are exchanged so each one stays on its own side. A sidewalk or a far curb does not. The front view and the bird's-eye view both draw the part of that curve the camera can see. A detection cache that stored lifted points instead of pixels is ignored and the detector is run again.

## Check

```bash
python scripts/test_tracks.py
```

No GPU. One projection check reads a local calibration file if that file is present.
