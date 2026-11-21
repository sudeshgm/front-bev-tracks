"""
Ultra-Fast-Lane-Detection (UFLD) video overlay (MIT).
Draws detected lane points.

Usage (on GPU):
  git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection
  # download a CULane or TuSimple checkpoint
  python ufld_overlay.py --video clip.mp4 --config Ultra-Fast-Lane-Detection/configs/culane.py --weights culane_18.pth --out overlay.mp4
"""
import argparse
import os
import sys
import cv2
import torch
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--ufld-root", default="Ultra-Fast-Lane-Detection")
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sys.path.insert(0, args.ufld_root)
    from utils.common import merge_config, get_model
    import utils.common as common

    # Minimal adapter: load config + model the way demo.py does
    # (exact import paths may need small adjustment on the GPU box)
    cfg = merge_config(args.config)
    model = get_model(cfg)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    # NOTE: UFLD expects a specific preprocessing (row anchors).
    # The GPU Bot should fill in the exact transform from the repo's demo.py
    # if the placeholder below fails.
    print("UFLD script skeleton ready. GPU Bot: complete the forward pass using the repo demo.")
    # Placeholder loop – GPU Bot will replace with real inference
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)  # identity until real inference is wired

    cap.release()
    writer.release()
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
