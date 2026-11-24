"""iter12: draw lanes (green) / main edges (orange; anchored = magenta, backfill = cyan, held = dashed-ish thin) of
several runs on the raw frame, side by side, each with a small BEV panel.
usage: overlay_runs.py --ds DS --clip munich --frames 165,565 --out DIR LABEL=RUN [LABEL=RUN ...]"""
import argparse, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import bev_smooth as BS
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--clip", required=True)
ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True); ap.add_argument("runs", nargs="+")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
runs = [r.split("=", 1) for r in a.runs]
P = {l: json.load(open(f"{r}/polylines/{a.clip}_polylines.json")) for l, r in runs}
d0 = next(iter(P.values())); K = d0["K"]
gm = BS.GroundModel(K["fx"], K["fy"], K["cx"], K["cy"], d0["cam_height_m"], d0["ground"]["pitch_rad"])
cap = cv2.VideoCapture(f"{a.ds}/{a.clip}.mp4")
want = sorted(int(v) for v in a.frames.split(","))
k = 0; ok = True
while ok and want:
    ok, bgr = cap.read()
    if not ok: break
    if k == want[0]:
        want.pop(0); panels = []
        for lab, _ in runs:
            fr = P[lab]["frames"][k]; im = bgr.copy()
            bev = np.zeros((im.shape[0], 260, 3), np.uint8)
            def bxy(x, z): return int(130 + x * 10), int(im.shape[0] - 20 - z * (im.shape[0] - 40) / 40.0)
            for kind, recs, polys in (("lane", fr["lanes"], fr["lanes_bev"]), ("edge", fr["edges"], fr["edges_bev"])):
                for r, p in zip(recs, polys):
                    p = np.asarray(p, float)
                    if len(p) < 2: continue
                    col = (80, 200, 80) if kind == "lane" else (0, 140, 255)
                    if r.get("anchored"): col = (255, 0, 255)
                    elif r.get("backfill"): col = (255, 255, 0)
                    th = 2 if r.get("held") and not r.get("anchored") else 4
                    uv = gm.project(p); uv = uv[np.isfinite(uv).all(1)]
                    if len(uv) >= 2:
                        cv2.polylines(im, [uv.astype(np.int32)], False, col, th)
                    cv2.polylines(bev, [np.array([bxy(x, z) for x, z in p], np.int32)], False, col, 2)
            for z in (10, 20, 30):
                cv2.putText(bev, f"{z}m", bxy(-12, z), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                cv2.line(bev, bxy(-12, z), bxy(12, z), (60, 60, 60), 1)
            cv2.putText(im, f"{lab} f{k}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            panels.append(np.hstack([im, bev]))
        out = np.vstack(panels)
        s = 1400.0 / out.shape[1]
        cv2.imwrite(f"{a.out}/{a.clip}_f{k:04d}.jpg", cv2.resize(out, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 85])
    k += 1
print("done", a.out)
