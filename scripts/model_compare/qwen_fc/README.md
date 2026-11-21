# Qwen-Drive FC-only perception

Uses only the VLM + perception head (no planner).
Other cameras in the 6-view ring are pure black images.

## Steps (GPU Bot)
1. Clone https://github.com/QwenLM/Qwen-Drive-1.0 (Apache-2.0)
2. Download Qwen/Qwen-Drive-1.0-4B and the perception/ subfolder
3. For each KITTI frame:
   ```
   python scripts/model_compare/qwen_fc/prepare_fc_ring.py \
       --fc-image frame.jpg \
       --out-frame /tmp/frame_fc
   ```
4. Run official:
   ```
   PYTHONPATH=src python scripts/run_perception.py \
       --vlm /path/to/Qwen-Drive-1.0-4B \
       --model /path/to/Qwen-Drive-1.0-4B/perception \
       --frames /tmp/frame_fc \
       --output outputs/qwen_fc
   ```
5. Visualize 3D boxes / BEV map projected back onto the front camera and write overlay videos under
   `testing_videos/model_compare/qwen_fc/`.

Free GPU only. Perception head is ~0.5 GB; VLM is the larger piece.
If memory is tight, report and stop.
