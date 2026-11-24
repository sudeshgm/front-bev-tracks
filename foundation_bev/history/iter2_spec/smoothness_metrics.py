#!/usr/bin/env python3
"""Smoothness metrics for BEV runs (iter1/iter2 foundation, or YOLO dumps in the same format).

Usage:
  python3 smoothness_metrics.py --run DIR [--run DIR2 ...] [--label NAME ...] [--out report.md]
  DIR must contain tracks/<clip>_tracks.csv and polylines/<clip>_polylines.json (iter1 or iter2 format).
  fps per clip: read from polylines JSON ("fps"), else run_meta.json in DIR, else --fps.

Per clip:
  tracks:  n_tracks, mean length, BEV acceleration RMS (m/s^2) of the drawn positions (bev_x, bev_z;
           constant ego speed cancels in the 2nd difference), the same for raw measurements (iter2 meas_x/z),
           probable ID switches per 100 frames (a track ends and a same-class track starts within
           10 frames and 3 m, i.e. fragments the re-link did not join).
  lines:   lateral jitter of nearest-left/right lane and edge at z=10/20 m (median |dx| per frame, m),
           presence %, and for iter2 the motion-compensated coefficient jitter per line id
           (median |dc0| m, |dc1|*10 m, |dc2|*100 m).
"""
import argparse, csv, glob, json, math, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_polylines import side_series, jitter  # noqa: E402


def load_tracks(path):
    tr = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            tr[(r["track_id"])].append(r)
    return tr


def accel_rms(rows, fps, kx, kz):
    acc = []
    rows = sorted(rows, key=lambda r: int(r["frame"]))
    pts = [(int(r["frame"]), r.get(kx, ""), r.get(kz, "")) for r in rows]
    pts = [(f, float(x), float(z)) for f, x, z in pts if x not in ("", None) and z not in ("", None)]
    dt = 1.0 / fps
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        if b[0] - a[0] == 1 and c[0] - b[0] == 1:
            ax = (c[1] - 2 * b[1] + a[1]) / dt / dt
            az = (c[2] - 2 * b[2] + a[2]) / dt / dt
            acc.append(ax * ax + az * az)
    return acc


def id_switch_proxy(tracks, gap=10, dist=3.0):
    ends, starts = [], []
    for tid, rows in tracks.items():
        rows = sorted(rows, key=lambda r: int(r["frame"]))
        cls = rows[0]["class"]
        f0, f1 = int(rows[0]["frame"]), int(rows[-1]["frame"])
        ends.append((f1, cls, float(rows[-1]["bev_x"]), float(rows[-1]["bev_z"]), tid))
        starts.append((f0, cls, float(rows[0]["bev_x"]), float(rows[0]["bev_z"]), tid))
    n = 0
    for fe, ce, xe, ze, te in ends:
        for fs, cs, xs, zs, ts in starts:
            if ts != te and cs == ce and 0 < fs - fe <= gap and math.hypot(xs - xe, zs - ze) <= dist:
                n += 1
                break
    return n


def shift(c, d):
    c0, c1, c2 = c
    return np.array([c0 + c1 * d + c2 * d * d, c1 + 2 * c2 * d, c2])


def coeff_jitter(frames, key, ds):
    by = defaultdict(list)
    for fr in frames:
        for l in fr.get(key, []):
            by[(l.get("side"), l["id"])].append((fr["frame"], np.array(l["c"], float)))
    d0, d1, d2 = [], [], []
    for _, seq in by.items():
        for (fa, ca), (fb, cb) in zip(seq, seq[1:]):
            if fb - fa != 1:
                continue
            p = shift(ca, ds)
            d = np.abs(cb - p)
            d0.append(d[0]); d1.append(d[1] * 10); d2.append(d[2] * 100)
    med = lambda v: round(float(np.median(v)), 3) if v else None  # noqa: E731
    return {"dc0_m": med(d0), "dc1x10_m": med(d1), "dc2x100_m": med(d2), "n": len(d0)}


def clip_metrics(run, clip, fps_default):
    pj = Path(run) / "polylines" / f"{clip}_polylines.json"
    tc = Path(run) / "tracks" / f"{clip}_tracks.csv"
    out = {"clip": clip}
    P = json.load(open(pj)) if pj.exists() else None
    fps = (P or {}).get("fps")
    if fps is None:
        rm = Path(run) / "run_meta.json"
        if rm.exists():
            fps = json.load(open(rm)).get("videos", {}).get(clip, {}).get("fps")
    fps = float(fps or fps_default)
    out["fps"] = fps
    nfr = len(P["frames"]) if P else None
    out["frames"] = nfr
    if tc.exists():
        T = load_tracks(tc)
        out["n_tracks"] = len(T)
        out["mean_track_len"] = round(float(np.mean([len(v) for v in T.values()])), 1) if T else 0
        acc = [a for rows in T.values() for a in accel_rms(rows, fps, "bev_x", "bev_z")]
        out["accel_rms_mps2"] = round(math.sqrt(np.mean(acc)), 3) if acc else None
        accm = [a for rows in T.values() for a in accel_rms(rows, fps, "meas_x", "meas_z")]
        out["accel_rms_raw_mps2"] = round(math.sqrt(np.mean(accm)), 3) if accm else None
        nsw = id_switch_proxy(T)
        out["id_switch_proxy_per100f"] = round(100.0 * nsw / max(nfr or 1, 1), 2) if T else None
    if P:
        fr = P["frames"]
        for kind in ("lanes", "edges"):
            for z in (10, 20):
                L, R = side_series(fr, f"{kind}_bev", z)
                for s, ser in (("L", L), ("R", R)):
                    med, p90, _ = jitter(ser)
                    out[f"{kind}_{s}{z}_jit"] = None if med is None else round(med, 3)
                    out[f"{kind}_{s}{z}_p90"] = None if p90 is None else round(p90, 3)
                    out[f"{kind}_{s}{z}_pres"] = round(100.0 * sum(x is not None for x in ser) / max(len(fr), 1), 1)
            if fr and kind in fr[0]:
                spd = P.get("ego_speed_mps", 0.0) if P.get("line_motion") == "straight" else 0.0
                out[f"{kind}_coef_jitter"] = coeff_jitter(fr, kind, spd / fps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--label", action="append")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    labels = a.label or [Path(r).name for r in a.run]
    allres = {}
    for run, lab in zip(a.run, labels):
        clips = sorted({Path(p).name.replace("_polylines.json", "") for p in glob.glob(f"{run}/polylines/*_polylines.json")} |
                       {Path(p).name.replace("_tracks.csv", "") for p in glob.glob(f"{run}/tracks/*_tracks.csv")})
        allres[lab] = [clip_metrics(run, c, a.fps) for c in clips]
    lines = ["# Smoothness metrics", "",
             "accel = RMS BEV acceleration of drawn track positions (m/s^2); idsw = probable ID switches /100 frames;",
             "jit = median |frame-to-frame change| of nearest left/right line lateral offset at z (m); pres = % frames present.", ""]
    cols = ["clip", "fps", "frames", "n_tracks", "accel_rms_mps2", "accel_rms_raw_mps2", "id_switch_proxy_per100f",
            "lanes_L10_jit", "lanes_R10_jit", "lanes_L20_jit", "lanes_R20_jit", "lanes_L10_pres", "lanes_R10_pres",
            "edges_L10_jit", "edges_R10_jit", "edges_L20_jit", "edges_R20_jit", "edges_L10_pres", "edges_R10_pres"]
    for lab, res in allres.items():
        lines += [f"## {lab}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in res:
            lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
        cj = [(r["clip"], r.get("lanes_coef_jitter"), r.get("edges_coef_jitter")) for r in res if r.get("lanes_coef_jitter")]
        if cj:
            lines += ["", "coefficient jitter (motion-compensated, median per frame): lanes / edges", ""]
            for c, l, e in cj:
                lines.append(f"- {c}: lanes {l}; edges {e}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
    if a.json:
        Path(a.json).write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
