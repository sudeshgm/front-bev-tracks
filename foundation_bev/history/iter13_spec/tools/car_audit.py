#!/usr/bin/env python3
"""iter13: which output main edges run inside a parked car's BEV footprint (lines_mt.car_inside: the same test the
iter13 rule uses), per edge kind: anchored / backfill / held_edgehold / held_tracker / kerb / live.
usage: car_audit.py --run RUN_DIR --dets DETS_DUMP_DIR [--json OUT.json]   (dets: the GPU run's dets_dump)"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import lines_mt as LM, bev_smooth as BS  # noqa: E402
CLIPS = ("kitti_highway_sample kitti_highway_exit_approach kitti_highway_late kitti_highway_final "
         "kitti_city_cyclist_candidate kitti_lane_split_candidate munich a2d2a a2d2b").split()
SYN = ("anchored", "backfill", "held_edgehold")


def kind(r):
    if r.get("anchored"):
        return "anchored"
    if r.get("backfill"):
        return "backfill"
    if r.get("held"):
        return "held_" + {"edgehold": "edgehold", "tracker": "tracker"}.get(str(r.get("hold_src")), "unknown")
    return "kerb" if r.get("kerb_occ") else "live"


def audit(run, dets):
    out = {}
    for clip in CLIPS:
        pj = f"{run}/polylines/{clip}_polylines.json"
        if not os.path.exists(pj):
            continue
        d = json.load(open(pj)); K = d["K"]
        gm = BS.GroundModel(K["fx"], K["fy"], K["cx"], K["cy"], d["cam_height_m"], d["ground"]["pitch_rad"])
        de = json.load(open(f"{dets}/{clip}_dets_ego.json"))["frames"]
        tot, hit = {}, {}
        for k, fr in enumerate(d["frames"]):
            cars = LM.car_footprints([o["box"] for o in de[k].get("objs") or [] if o.get("cls") == "car"], gm.lift)
            for r, p in zip(fr["edges"], fr["edges_bev"]):
                if r.get("role") not in ("main_L", "main_R") or len(p) < 2:
                    continue
                kd = kind(r); tot[kd] = tot.get(kd, 0) + 1
                p = np.asarray(p, float); x = np.full(len(LM.ZS), np.nan); o = np.argsort(p[:, 1])
                m = (LM.ZS >= p[:, 1].min()) & (LM.ZS <= p[:, 1].max()); x[m] = np.interp(LM.ZS[m], p[o, 1], p[o, 0])
                if LM.car_inside(x, -1 if r["role"] == "main_L" else 1, cars):
                    hit.setdefault(kd, []).append(f"{r['role'][-1]}{int(fr['frame'])}")
        out[clip] = {"edges": tot, "inside_car": hit, "synthetic_inside": sum(len(hit.get(s, [])) for s in SYN)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--dets", required=True)
    ap.add_argument("--json"); a = ap.parse_args()
    res = audit(a.run, a.dets)
    for c, r in res.items():
        print(f"{c}: synthetic (anchored/backfill/edgehold) inside a car: {r['synthetic_inside']}  "
              + " ".join(f"{k}={len(v)}[{' '.join(v[:12])}{' ...' if len(v) > 12 else ''}]" for k, v in sorted(r["inside_car"].items())))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
