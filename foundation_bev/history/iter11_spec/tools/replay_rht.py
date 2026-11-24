"""iter9 offline check of the RHT dot rule on a run's tracks (render-only rule; tracks.csv bev_x/bev_z = ego-relative
smoothed positions). usage: replay_rht.py RUN EGO_RUN [clip ...]  (EGO_RUN/polylines/<clip>_polylines.json 'ego' gives the
selected travel s). Prints, per clip, how many drawn-trail track-frames the iter8 rules leave and how many the iter9
RHT rule turns into dots (with the track ids)."""
import csv, glob, json, math, os, sys
import numpy as np
run, egor = sys.argv[1], sys.argv[2]; fps = 10.0
clips = sys.argv[3:] or sorted(os.path.basename(p)[:-11] for p in glob.glob(f"{run}/tracks/*_tracks.csv"))
for clip in clips:
    eg = json.load(open(f"{egor}/polylines/{clip}_polylines.json"))["frames"]
    ex = np.array([(e.get("ego") or {}).get("x", 0) for e in eg]); ez = np.array([(e.get("ego") or {}).get("z", 0) for e in eg])
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(ex), np.diff(ez)))]); n = len(s)
    T = {}
    for r in csv.DictReader(open(f"{run}/tracks/{clip}_tracks.csv")):
        T.setdefault(int(r["track_id"]), []).append((int(r["frame"]), float(r["bev_x"]), float(r["bev_z"])))
    trails = dots = 0; ids = set()
    for now in range(n):
        k0 = max(1, now - 10); v_ego = float(np.median(np.diff(s[k0 - 1:now + 1]))) * fps if now >= 1 else 0.0
        for tid, rows in T.items():
            h = [r for r in rows if r[0] <= now]
            if not h or h[-1][0] != now or len(h) < 5: continue
            fr = np.array([r[0] for r in h]); wx = np.array([r[1] for r in h]); wz = np.array([r[2] + s[r[0]] for r in h])
            dt = max((fr[-1] - fr[0]) / fps, 1e-3)
            dzw = (np.median(wz[-3:]) - np.median(wz[:3])) / dt; v_g = abs(dzw); v_lat = abs(np.median(wx[-3:]) - np.median(wx[:3])) / dt
            if v_g < 1.5 or (v_lat > v_g and v_lat > 1.0): continue  # iter8: already a dot
            if not (0 <= h[-1][2] <= 45 and abs(h[-1][1]) <= 15): continue
            trails += 1
            if h[0][1] > 1.5 and dzw < 0 and v_lat < v_g and v_g <= 1.5 * max(v_ego, 0.5):
                dots += 1; ids.add(tid)
    print(f"{clip}: iter8 trail track-frames {trails}, iter9 RHT -> dots {dots}, tracks {sorted(ids)}")
