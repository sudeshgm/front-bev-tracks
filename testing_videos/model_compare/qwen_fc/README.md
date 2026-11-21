# Qwen-Drive-1.0 perception, front camera only (FC + 5 black cameras)

- Model: Qwen/Qwen-Drive-1.0-4B VLM feature taps + perception/ BEV head (Apache-2.0). No planner.
- Code: QwenLM/Qwen-Drive-1.0 @28091c1 (official loop of scripts/run_perception.py) via
  front-bev-tracks qwen-fc-results: scripts/model_compare/qwen_fc/prepare_fc_ring.py (fixed),
  run_perception_t4.py (T4 shim: fp16, SDPA, VLM cuda:0 / head cuda:1, fp16-safe deformable attention; no quantization),
  project_overlay.py (projection back onto the clip frame, CPU).
- Ring: KITTI frame un-squeezed to 1242x375, letterboxed to 896:512, as CAM_FRONT with KITTI P2 intrinsics and a
  1.65 m camera pose; the other five nuScenes cameras are black with the official nuScenes demo calibration.
- Overlay: left = clip frame with 3D boxes (score >= 0.25) and predicted BEV map on the ground plane
  (blue driveable, yellow road line, red road edge, teal crosswalk, green walkway); right = BEV panel, forward up,
  all boxes incl. those placed behind the car. Videos run at 2 fps when only a subset of frames was processed.
- Compute: free Kaggle (hiddenslave, private kernel kitti-fc-qwen-perception). $0 paid.

## Summary
```
run_meta: rc=0 mode=all wall_s=1900 infer_stage_s=1767 fails= fbt=5f76337172b40edd2d498ea4f9d599bad8292a27 qd=28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0
kitti_city_cyclist_candidate: 30 frames, 6.72 s/frame, load 111.6 s, peak VRAM {'cuda:0': 10.78, 'cuda:1': 5.88}, boxes>=0.25: 78 total, 78 in FC view, 0 behind ego (black cams)
kitti_highway_exit_approach: 40 frames, 6.06 s/frame, load 55.6 s, peak VRAM {'cuda:0': 10.78, 'cuda:1': 5.88}, boxes>=0.25: 1 total, 1 in FC view, 0 behind ego (black cams)
kitti_highway_final: 30 frames, 6.06 s/frame, load 54.3 s, peak VRAM {'cuda:0': 10.78, 'cuda:1': 5.88}, boxes>=0.25: 5 total, 5 in FC view, 0 behind ego (black cams)
kitti_highway_late: 40 frames, 6.09 s/frame, load 55.6 s, peak VRAM {'cuda:0': 10.78, 'cuda:1': 5.88}, boxes>=0.25: 5 total, 5 in FC view, 0 behind ego (black cams)
kitti_highway_sample: 40 frames, 6.07 s/frame, load 54.4 s, peak VRAM {'cuda:0': 10.78, 'cuda:1': 5.88}, boxes>=0.25: 3 total, 3 in FC view, 0 behind ego (black cams)
kitti_lane_split_candidate: 30 frames, 6.06 s/frame, load 56.1 s, peak VRAM {'cuda:0': 10.78, 'cuda:1': 5.88}, boxes>=0.25: 607 total, 607 in FC view, 0 behind ego (black cams)
```
