"""
TwinLiteNet inference on video (MIT license).
Produces overlay with drivable area (green) + lane lines (red).

Usage (on GPU machine):
  git clone https://github.com/chequanghuy/TwinLiteNet
  # download pretrained/best.pth into TwinLiteNet/
  python twinlite_overlay.py --video /path/to/clip.mp4 --weights TwinLiteNet/pretrained/best.pth --out overlay.mp4
"""
import argparse
import os
import sys
import cv2
import torch
import numpy as np

def load_model(weights_path, twinlite_root):
    sys.path.insert(0, twinlite_root)
    from model import TwinLite as net
    model = net.TwinLiteNet()
    state = torch.load(weights_path, map_location="cpu")
    # handle DataParallel wrapper
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def preprocess(frame, size=(640, 360)):
    img = cv2.resize(frame, size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img = np.ascontiguousarray(img)
    t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
    if torch.cuda.is_available():
        t = t.cuda()
    return t, size

@torch.no_grad()
def run_frame(model, tensor):
    da_out, ll_out = model(tensor)
    da = torch.argmax(da_out, dim=1)[0].cpu().numpy()
    ll = torch.argmax(ll_out, dim=1)[0].cpu().numpy()
    return da, ll

def overlay(frame, da, ll, size):
    h, w = frame.shape[:2]
    da = cv2.resize(da.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    ll = cv2.resize(ll.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    out[da > 0] = (0, 180, 0)          # green drivable
    out[ll > 0] = (0, 0, 255)          # red lanes
    # blend
    return cv2.addWeighted(frame, 0.6, out, 0.4, 0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--twinlite-root", default="TwinLiteNet")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = load_model(args.weights, args.twinlite_root)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tensor, _ = preprocess(frame)
        da, ll = run_frame(model, tensor)
        vis = overlay(frame, da, ll, None)
        writer.write(vis)

    cap.release()
    writer.release()
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
