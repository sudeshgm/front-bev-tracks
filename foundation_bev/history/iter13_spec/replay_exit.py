"""Offline iter4 replay on an iter2 clip: classical BEV masks (bev_ipm) -> lines_mt tracking -> before/after
stills (front with projected iter4 lines | iter2 BEV | iter4 BEV). No models are run."""
import sys, os, json, argparse
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, "/workspace/foundation_bev/iter3_spec/src")
import bev_smooth as BS, bev_ipm as IP, lines_mt as LM

COL_LANE, COL_EDGE, COL_SPLIT, COL_GORE = (80, 210, 90), (40, 140, 255), (40, 200, 255), (60, 110, 200)
COL_CAR = (60, 60, 255)


def bev_canvas(h, w=400, max_d=40.0):
    sc = min((h - 30) / max_d, (w / 2 - 10) / 15.0)
    p = np.full((h, w, 3), 22, np.uint8)
    for m in (10, 20, 30, 40):
        y = int(h - 20 - m * sc); cv2.line(p, (0, y), (10, y), (80, 80, 80), 1)
        cv2.putText(p, f"{m}m", (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1, cv2.LINE_AA)
    return p, sc


def to_px(P, h, w, sc):
    return np.stack([w / 2 + P[:, 0] * sc, h - 20 - P[:, 1] * sc], 1).round().astype(np.int32)


def draw_iter4(h, lanes, edges, gores, events_now, w=400):
    p, sc = bev_canvas(h, w)
    for g in gores:  # subtle: outline + sparse hatch
        poly = to_px(np.asarray(g["poly"]), h, w, sc)
        ov = p.copy(); cv2.fillPoly(ov, [poly], COL_GORE); p = cv2.addWeighted(ov, 0.25, p, 0.75, 0)
        cv2.polylines(p, [poly], True, COL_EDGE, 1, cv2.LINE_AA)
    for e in edges:
        P = LM.curve_points(e["x"])
        if len(P) >= 2:
            cv2.polylines(p, [to_px(P, h, w, sc)], False, COL_SPLIT if e.get("split") else COL_EDGE, 2, cv2.LINE_AA)
    for l in lanes:
        P = LM.curve_points(l["x"])
        if len(P) >= 2:
            cv2.polylines(p, [to_px(P, h, w, sc)], False, COL_LANE, 2, cv2.LINE_AA)
            if l.get("parent") is not None and l.get("fork_z") is not None:  # fork node
                k = int(round((l["fork_z"] - LM.ZS[0]) / 0.5))
                if 0 <= k < len(LM.ZS) and np.isfinite(l["x"][k]):
                    c = to_px(np.array([[l["x"][k], l["fork_z"]]]), h, w, sc)[0]
                    cv2.circle(p, tuple(int(v) for v in c), 4, (255, 255, 255), 1, cv2.LINE_AA)
    ego = to_px(np.array([[0, 2.0], [-0.9, -0.5], [0.9, -0.5]]), h, w, sc)
    cv2.fillConvexPoly(p, ego, (230, 230, 230))
    cv2.putText(p, "iter4 replay (classical masks)", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return p


def project_draw(img, gm, x, col, th=2):
    P = LM.curve_points(x)
    if len(P) < 2:
        return
    uv = gm.project(P)
    ok = np.isfinite(uv).all(1)
    if ok.sum() >= 2:
        cv2.polylines(img, [uv[ok].round().astype(np.int32)], False, col, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="kitti_highway_exit_approach")
    ap.add_argument("--frames", default="8,19,20")
    ap.add_argument("--out", default=os.path.join(HERE, "replay"))
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    want = {int(v) for v in a.frames.split(",")}
    os.makedirs(a.out, exist_ok=True)
    m = json.load(open("/workspace/foundation_bev/iter2/run_meta.json"))["videos"][a.clip]
    fx, fy, cx, cy = m["K"]
    gm = BS.GroundModel(fx, fy, cx, cy, m["cam_height"], m["ground"]["pitch_rad"])
    Hm = IP.ipm_homography(gm)
    cap = cv2.VideoCapture(f"/workspace/gpubot/foundation-bev-23nov/ds/{a.clip}.mp4")
    lt = LM.MultiLineTracker("lane"); gt = LM.GoreTracker()
    # one tracker per edge role: main left / main right boundary of the drivable area, and the split pair
    etrs = {r: LM.MultiLineTracker("edge", n_birth=3, n_death=4, gate_max=2.0, alpha=0.3) for r in ("main_L", "main_R", "split_in", "split_out")}
    stats = {"frames": 0, "lanes_per_frame": [], "edges_per_frame": [], "gore_frames": 0, "split_edge_frames": 0}
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        b = IP.warp(f, Hm); g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        val = IP.valid_mask(f.shape, Hm, gm.horizon_v)
        paint0 = IP.paint_mask(g, val)
        road = IP.asphalt_mask(b, val, paint=paint0)
        road_d = cv2.dilate(road.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        paint = paint0 & road_d
        road_f = cv2.morphologyEx((road | paint).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        comps = LM.components(paint, IP.RES)
        chains, stripes = LM.extract_chains(comps)
        obs = [LM.smooth_chain(c) for c in chains]
        lanes = lt.step(k, obs)
        ed = LM.edge_chains(road_f, IP.RES, valid=cv2.erode(val.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        edges = []
        for r, trr in etrs.items():
            for e in trr.step(k, [ed[r]]):
                e["split"] = r.startswith("split"); e["role"] = r
                edges.append(e)
        if np.isfinite(ed["split_in"]).any():
            stats["split_edge_frames"] += 1
        gores = gt.step(LM.gore_polygons(stripes))
        stats["frames"] += 1; stats["lanes_per_frame"].append(len(lanes)); stats["edges_per_frame"].append(len(edges))
        stats["gore_frames"] += int(bool(gores))
        if k in want:
            h = f.shape[0]
            front = f.copy()
            for e in edges:
                project_draw(front, gm, e["x"], COL_SPLIT if e["split"] else COL_EDGE)
            for l in lanes:
                project_draw(front, gm, l["x"], COL_LANE)
            for gg in gores:
                P = np.asarray(gg["poly"]); uv = gm.project(P); okp = np.isfinite(uv).all(1)
                if okp.sum() >= 3:
                    cv2.polylines(front, [uv[okp].round().astype(np.int32)], True, COL_EDGE, 1, cv2.LINE_AA)
            after = draw_iter4(h, lanes, edges, gores, [])
            st = f"/workspace/foundation_bev/iter2/stills/{a.clip}_f{k:04d}.png"
            before = cv2.imread(st)[:, -400:] if os.path.exists(st) else np.zeros_like(after)
            cv2.putText(before, "iter2 (before)", (60, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            dbg = cv2.resize(b, (400, int(b.shape[0] * 400 / b.shape[1])))
            out = np.hstack([front, before, after])
            cv2.imwrite(os.path.join(a.out, f"{a.clip}_f{k:04d}_before_after{a.suffix}.png"), out)
            vis = b.copy(); vis[road_f] = (0.6 * vis[road_f] + [0, 50, 0]).astype(np.uint8); vis[paint] = (0, 0, 255)
            cv2.imwrite(os.path.join(a.out, f"{a.clip}_f{k:04d}_bevmask.png"), vis)
        k += 1
    stats["events"] = lt.events
    stats["n_lane_ids"] = len(lt.hist); stats["n_edge_ids"] = {r: len(t.hist) for r, t in etrs.items()}
    stats["forks"] = [e for e in lt.events if e["type"] == "fork"]
    json.dump(stats, open(os.path.join(a.out, f"{a.clip}_replay_stats.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("events",)}, default=float))


if __name__ == "__main__":
    main()
