"""Offline replay: iter2 tracks.csv + polylines -> iter3 track filters (score, min hits, bicycle rider,
off-road VRU on highways, static flag). Reports what would be dropped / drawn static and the accel RMS
of what remains drawn as moving tails. Detection-level gates (ground contact, vertical plane) need boxes
and depth, which iter2 did not dump, so they are only unit-tested here."""
import sys, os, csv, json, math, argparse
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import bev_gate as G
import bev_smooth as BS

HIGHWAY = {"kitti_highway_sample", "kitti_highway_exit_approach", "kitti_highway_late", "kitti_highway_final"}


def accel(rows, fps):
    acc = []
    for a, b, c in zip(rows, rows[1:], rows[2:]):
        if b[0] - a[0] == 1 and c[0] - b[0] == 1:
            ax = (c[1] - 2 * b[1] + a[1]) * fps * fps; az = (c[2] - 2 * b[2] + a[2]) * fps * fps
            acc.append(ax * ax + az * az)
    return acc


def edge_fn(fr):
    L = [e for e in fr.get("edges", []) if e.get("side", 0) < 0]
    R = [e for e in fr.get("edges", []) if e.get("side", 0) > 0]
    def f(z):
        lo = BS.x_at(L[0]["c"], z) if L else None
        hi = BS.x_at(R[0]["c"], z) if R else None
        return lo, hi
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/workspace/foundation_bev/iter2")
    ap.add_argument("--ego", default="", help="npy dict of visual ego-motion per clip (optional)")
    ap.add_argument("--ego-mode", default="prior", choices=["prior", "scene"])
    ap.add_argument("--out", default="replay_iter2_tracks.md")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.run, "run_meta.json")))["videos"]
    vo = np.load(a.ego, allow_pickle=True).item() if a.ego else {}
    rows_md = ["| clip | tracks | drawn moving | drawn static | drop short | drop score | drop bicycle no rider | drop off-road VRU | accel RMS iter2 (all) | accel RMS iter3 (moving) |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    allres = {}
    scene_sp = {}
    for clip in sorted(meta):
        p = os.path.join(a.run, "tracks", f"{clip}_tracks.csv")
        if not os.path.exists(p):
            continue
        m = meta[clip]; fps = m["fps"]; speed = m["ego_speed_mps"]
        T = defaultdict(list)
        for r in csv.DictReader(open(p)):
            T[r["track_id"]].append(r)
        nF = max(int(r["frame"]) for rs in T.values() for r in rs) + 1
        if clip in vo:
            e = {k: np.asarray(v) for k, v in vo[clip].items() if isinstance(v, list)}
        else:
            e = {"s": np.arange(nF) * speed / fps}
        tracks = {}
        for tid, rs in T.items():
            rs.sort(key=lambda r: int(r["frame"]))
            hits = [r for r in rs if r["src"] != "coast"]
            tracks[tid] = {"cls": Counter(r["class"] for r in rs).most_common(1)[0][0],
                           "frames": [int(r["frame"]) for r in rs], "x": [float(r["bev_x"]) for r in rs],
                           "z": [float(r["bev_z"]) for r in rs], "scores": [float(r["score"]) for r in hits] or [0.0],
                           "hits": len(hits)}
        if a.ego_mode == "scene" and clip not in vo:
            e = G.scene_ego_profile(tracks, nF, fps, speed)
            scene_sp[clip] = {"median": round(float(np.median(e["speed"])), 1), "p10": round(float(np.percentile(e["speed"], 10)), 1),
                              "p90": round(float(np.percentile(e["speed"], 90)), 1), "valid_frac": round(e["valid_frac"], 2), "prior": speed}
        polyp = os.path.join(a.run, "polylines", f"{clip}_polylines.json")
        edges = {fr["frame"]: edge_fn(fr) for fr in json.load(open(polyp))["frames"]} if os.path.exists(polyp) else None
        dec = G.track_filter(tracks, fps, e, edges, highway=clip in HIGHWAY)
        c = Counter(dec.values())
        a2 = [v for t in tracks.values() for v in accel(list(zip(t["frames"], t["x"], t["z"])), fps)]
        a3 = [v for tid, t in tracks.items() if dec[tid] == "draw" for v in accel(list(zip(t["frames"], t["x"], t["z"])), fps)]
        f = lambda v: f"{math.sqrt(np.mean(v)):.2f}" if v else "–"
        rows_md.append(f"| {clip} | {len(tracks)} | {c['draw']} | {c['draw_static']} | {c['drop_short']} | {c['drop_score']} | "
                       f"{c['drop_bicycle_no_rider']} | {c['drop_offroad_vru']} | {f(a2)} | {f(a3)} |")
        allres[clip] = {"n": len(tracks), **dict(c), "dropped_by_class": dict(Counter(tracks[t]["cls"] for t, d in dec.items() if d.startswith("drop"))),
                        "decisions": dec}
    md = "# iter2 tracks replayed through iter3 track filters\n\nego motion: " + ("visual (" + a.ego + ")" if a.ego else a.ego_mode + " speed, straight") + "\n\nscene ego speed (speed, n tracks, accepted, prior): " + json.dumps(scene_sp) + "\n\n" + "\n".join(rows_md) + "\n"
    open(a.out, "w").write(md)
    json.dump(allres, open(a.out.replace(".md", ".json"), "w"), indent=1)
    print(md)
    for k, v in allres.items():
        print(k, "dropped by class", v["dropped_by_class"])


if __name__ == "__main__":
    main()
