"""CPU check of the iter4 ego-speed fix on all 9 clips (classical IPM phase correlation, no models)."""
import sys, os, json, cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "src"))
import bev_smooth as BS, ego_motion as EM
cams = json.load(open(os.path.join(HERE, "cams_iter4.json")))
meta = json.load(open("/workspace/foundation_bev/iter3/run_meta.json"))["videos"]
it3 = {r.split(" | ")[0]: r for r in json.load(open("/workspace/foundation_bev/iter3/run_meta.json")).get("ego_speed_table", [])}
rows = ["| clip | file fps | capture hz | repeats | prior m/s | iter3 flow m/s | iter3 scene m/s | iter4 BEV phase-corr m/s (valid) |", "|---|---|---|---|---|---|---|---|"]
for clip in sorted(meta):
    m = meta[clip]; fx, fy, cx, cy = m["K"]; cam = cams[clip]
    gm = BS.GroundModel(fx, fy, cx, cy, m["cam_height"], m["ground"]["pitch_rad"])
    z0 = 12.0 if clip == "munich" else 7.0
    est = EM.BevPhaseEgo(gm, cam["capture_hz"], m["ego_speed_mps"], z_band=(z0, 30.0))
    cap = cv2.VideoCapture(f"/workspace/gpubot/foundation-bev-23nov/ds/{clip}.mp4"); n = 0
    while n < 120:
        ok, f = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV); hud = (hsv[..., 1] > 90) & (hsv[..., 2] > 80) if clip.startswith("a2d2") else None
        est.step(f, hud); n += 1
    r = est.finalize(n)
    p3 = it3.get(clip, "").split(" | ")
    rows.append(f"| {clip} | {m['fps']} | {cam['capture_hz']} | {r['n_repeat']} | {m['ego_speed_mps']} | {p3[2] if len(p3) > 2 else ''} | {p3[3] if len(p3) > 3 else ''} | {r['speed_median_mps']:.1f} ({r['valid_frac']:.2f}) |")
open(os.path.join(HERE, "ego_speed_table.md"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
