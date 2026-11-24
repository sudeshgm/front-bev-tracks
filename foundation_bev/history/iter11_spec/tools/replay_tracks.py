"""iter8 offline replay of the object smoother on GPU tracks.csv (association fixed as on GPU).
For every track: rows with a measurement -> ObjTracker._refilter + _rts (the same code path finalize() uses),
stationary pin, then the metric's accel_rms. Lets KF changes be A/B-tested on real stage-1 measurements.
usage: replay_tracks.py RUN_DIR OUT_DIR CLIPS [variant]   (variant only labels; behaviour comes from env/knobs)"""
import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import bev_smooth as BS

RUN, OUT, CLIPS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
meta = json.load(open(f"{RUN}/run_meta.json"))["videos"]
os.makedirs(f"{OUT}/tracks", exist_ok=True); os.makedirs(f"{OUT}/polylines", exist_ok=True)
for clip in CLIPS:
    v = meta[clip]; fps = float(v.get("fps") or 10.0); es = v["ego_speed"]
    prior = float(es["prior_mps"]); sel = float(es["speed_median_mps"]); src_sel = es["source"]
    rows = list(csv.DictReader(open(f"{RUN}/tracks/{clip}_tracks.csv")))
    by = {}
    for r in rows:
        by.setdefault(r["track_id"], []).append(r)
    T = BS.ObjTracker(fps, prior)
    out = []
    for tid, rr in by.items():
        rr.sort(key=lambda r: int(r["frame"]))
        hist = [dict(frame=int(r["frame"]), meas=(float(r["meas_x"]), float(r["meas_z"]) + prior * int(r["frame"]) / fps),
                     z_ego=float(r["meas_z"]), src=r["src"], score=float(r["score"]), box=None, cls=r["class"])
                for r in rr if r["meas_x"] not in ("", None)]
        if len(hist) < 2:
            continue
        h2, xs, stationary = T._smooth_final(hist, rr[0]["class"])
        pos = xs[:, :2].copy()
        if stationary:
            pos[:] = np.median(pos, 0)
        frames = np.array([h["frame"] for h in h2])
        if not stationary and BS.REPIN and src_sel != "prior":
            pos, stationary = BS.repin_true_world(pos, xs[:, 2:], frames, fps, prior, sel)
        for h, p in zip(h2, pos):
            out.append({"frame": h["frame"], "track_id": tid, "class": rr[0]["class"], "score": 0,
                        "bev_x": p[0], "bev_z": p[1] - prior * h["frame"] / fps, "filt_x": 0, "filt_z": 0,
                        "meas_x": h["meas"][0] if h["meas"] else "", "meas_z": (h["meas"][1] - prior * h["frame"] / fps) if h["meas"] else "",
                        "coasted": int(h["meas"] is None), "src": h["src"], "stationary": int(stationary), "draw": "draw"})
    out.sort(key=lambda r: (r["frame"], int(r["track_id"])))
    with open(f"{OUT}/tracks/{clip}_tracks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    p = f"{RUN}/polylines/{clip}_polylines.json"
    if os.path.exists(p) and not os.path.exists(f"{OUT}/polylines/{clip}_polylines.json"):
        os.symlink(p, f"{OUT}/polylines/{clip}_polylines.json")
