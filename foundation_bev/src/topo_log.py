"""iter15 (diagnostic only, FBEV_TOPO_LOG=<path.jsonl>; never changes outputs): per-frame road-mask topology log.

For the pass-1 drivable mask (filled SAM road after the VPLANE gate) and the same mask with *confirmed* WildDet3D vehicle
boxes filled back in as road, take the 8-connected road component under the ego corridor and log every row (step 2 px)
where that component has >= 2 runs: the gaps between runs are candidate gore islands / fork wedges *inside* the road.
A vehicle box is confirmed when: class car/truck, score >= 0.35, bottom below the horizon, ground width 1.2-3.6 m at its
bottom row (initial ground model), and road support >= 0.25 in a band just below its bottom edge."""
import json
import numpy as np
import cv2

VEH = ("car", "truck")


def _confirm(objs, road, gm, h, w):
    hv = gm.horizon_v; out = []
    for o in objs:
        if o.get("cls") not in VEH:
            continue
        x0, y0, x1, y1 = [float(v) for v in o["box"][:4]]
        rec = {"box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)], "s": round(float(o["score"]), 3), "ok": 0}
        if o["score"] < 0.35 or y1 <= hv + 4:
            out.append(rec); continue
        p = gm.lift([x0, x1], [y1, y1])
        if not np.all(np.isfinite(p)):
            out.append(rec); continue
        wm = float(p[1, 0] - p[0, 0]); rec["wm"] = round(wm, 2); rec["z"] = round(float(p[0, 1]), 1)
        a, b = max(0, int(x0)), min(w, int(np.ceil(x1)))
        ya, yb = min(h - 1, int(y1)), min(h, int(y1) + max(3, int(0.08 * (y1 - y0)) + 3))
        band = road[ya:yb, a:b] if b > a and yb > ya else np.zeros(0)
        sup = float((band > 0).mean()) if band.size else 0.0; rec["sup"] = round(sup, 2)
        rec["ok"] = int(1.2 <= wm <= 3.6 and sup >= 0.25)
        out.append(rec)
    return out


def _ego_component(m, u_ego, hv, h):
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    for v in range(h - 1, int(hv) + 6, -1):  # lowest row with road near the ego column
        row = lab[v]; idx = np.flatnonzero(row > 0)
        if idx.size:
            j = idx[np.argmin(np.abs(idx - u_ego))]
            if abs(j - u_ego) < 0.25 * m.shape[1]:
                return lab == row[j]
    return None


def _gaps(comp, depth, gm, hv, h):
    rows = []
    if comp is None:
        return rows
    for v in range(int(hv) + 6, h, 2):
        r = comp[v].astype(np.int8); d = np.diff(np.concatenate([[0], r, [0]]))
        s = np.flatnonzero(d == 1); e = np.flatnonzero(d == -1)
        if len(s) < 2:
            continue
        zr = gm.lift([gm.cx], [v])[0, 1]
        gl = []
        for i in range(len(s) - 1):
            u0, u1 = int(e[i]), int(s[i + 1])
            rf = None
            if depth is not None and np.isfinite(zr) and u1 > u0:
                dd = depth[v, u0:u1].astype(np.float32); dd = dd[np.isfinite(dd) & (dd > 0)]
                rf = round(float((dd < 0.85 * zr).mean()), 2) if dd.size else None
            gl.append([u0, u1, rf])
        rows.append([v, int(s[0]), int(e[-1]), gl])  # row, outer left, outer right, gaps [u0, u1, raised]
    return rows


def log_frame(path, tag, frame_i, road, objs, depth, gm, u_ego, h, w):
    try:
        m = (np.asarray(road) > 0)
        veh = _confirm(objs, m, gm, h, w)
        mf = m.copy()
        for rec in veh:
            if rec["ok"]:
                x0, y0, x1, y1 = rec["box"]
                mf[max(0, int(y0)):min(h, int(np.ceil(y1)) + 2), max(0, int(x0)):min(w, int(np.ceil(x1)))] = True
        hv = gm.horizon_v
        out = {"tag": tag, "f": int(frame_i), "hw": [int(h), int(w)], "u_ego": float(u_ego), "hv0": float(hv),
               "pitch0": float(gm.pitch), "veh": veh,
               "raw": _gaps(_ego_component(m, u_ego, hv, h), depth, gm, hv, h),
               "fill": _gaps(_ego_component(mf, u_ego, hv, h), depth, gm, hv, h)}
        with open(path, "a") as fh:
            fh.write(json.dumps(out, separators=(",", ":")) + "\n")
    except Exception as ex:  # diagnostics must never break a run
        with open(path, "a") as fh:
            fh.write(json.dumps({"tag": tag, "f": int(frame_i), "err": repr(ex)}) + "\n")
