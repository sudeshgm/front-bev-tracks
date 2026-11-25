"""iter15: ego speed / yaw / trajectory error of pipeline runs against a reference (KITTI OXTS: public GT; fc1 VO:
private reference, local only). Reads each run's results/<clip>.json (selected speed) and polylines (exported ego
x, z, psi; psi + = right). Variants per clip: before (iter14 run), signfix (iter14 psi negated: sign fix only), after.
Reference path: ds = v/hz, psi = cumsum(yaw_right/hz), x = cumsum(ds sin psi), z = cumsum(ds cos psi) (same integration).
Prints aggregate numbers only. usage: ego_eval.py OUT_JSON CLIP:BEFORE_DIR:AFTER_DIR:REF [...]
REF = oxts:<oxts.json>:<start> | sub:<kitti_oxts_subset.json>:<clip> | vo:<ego_motion_4fps.csv>"""
import sys, json, csv
import numpy as np

def ref_series(ref, n):
    p = ref.split(":")
    if p[0] == "sub":
        o = json.load(open(p[1]))["clips"][p[2]]
        return np.array(o["vf"][:n], float), -np.degrees(np.array(o["wz"][:n], float))
    if p[0] == "oxts":
        o = json.load(open(p[1])); a = int(p[2])
        return np.array(o["vf"][a:a + n], float), -np.degrees(np.array(o["wz"][a:a + n], float))
    R = list(csv.DictReader(open(p[1])))[:n]
    v = np.array([float(r["speed_mps"]) if r["speed_mps"] else np.nan for r in R])
    y = np.array([-np.degrees(float(r["yaw_rate_rps"])) if r["yaw_rate_rps"] else np.nan for r in R])
    return v, y

def run_series(d, clip, n, neg=False):
    P = json.load(open(f"{d}/polylines/{clip}_polylines.json")); hz = float(P.get("fps") or 10)
    E = [f.get("ego") or {} for f in P["frames"]][:n]
    psi = np.array([e.get("psi", np.nan) for e in E]); x = np.array([e.get("x", np.nan) for e in E]); z = np.array([e.get("z", np.nan) for e in E])
    if neg:
        psi = -psi; x = -x
    sp = json.load(open(f"{d}/results/{clip}.json"))["ego_speed"]["speed_median_mps"]
    yaw = np.degrees(np.concatenate([[0.0], np.diff(psi)])) * hz
    return {"speed": sp, "yaw": yaw, "x": x, "z": z, "hz": hz}

def integ(v, y, hz):
    v = np.where(np.isfinite(v), v, np.nanmedian(v)); y = np.where(np.isfinite(y), y, 0.0)
    ds = np.concatenate([[0.0], v[1:] / hz]); psi = np.cumsum(np.concatenate([[0.0], np.radians(y[1:]) / hz]))
    return np.cumsum(ds * np.sin(psi)), np.cumsum(ds * np.cos(psi))

def score(r, v, y):
    hz = r["hz"]; m = np.isfinite(y) & np.isfinite(r["yaw"]); m[0] = False
    e, t = r["yaw"][m], y[m]; err = e - t
    corr = float(np.corrcoef(e, t)[0, 1]) if e.std() > 1e-9 and t.std() > 1e-9 else float("nan")
    vm = float(np.nanmedian(v[1:])); xr, zr = integ(v, y, hz)
    k = np.isfinite(r["x"]) & np.isfinite(r["z"])
    pe = np.hypot(r["x"][k] - xr[k], r["z"][k] - zr[k]); last = np.where(k)[0][-1]
    yr2 = r["yaw"][1:]
    return {"speed": round(r["speed"], 2), "speed_ref": round(vm, 2), "speed_err_pct": round(100 * (r["speed"] / vm - 1), 1),
            "speed_frame_mae": round(float(np.nanmedian(np.abs(r["speed"] - v[1:]))), 2),
            "yaw_corr": round(corr, 2), "yaw_mae_dps": round(float(np.median(np.abs(err))), 2),
            "yaw_rmse_dps": round(float(np.sqrt(np.mean(err ** 2))), 2), "heading_drift_deg": round(float(np.sum(err) / hz), 1),
            "path_end_err_m": round(float(pe[-1]), 2), "path_lat_end_err_m": round(float(abs(r["x"][last] - xr[last])), 2),
            "path_rms_err_m": round(float(np.sqrt(np.mean(pe ** 2))), 2), "path_len_ref_m": round(float(np.hypot(xr[last], zr[last])), 1),
            "yaw_jerk_dps2": round(float(np.sqrt(np.mean((np.diff(yr2) * hz) ** 2))), 2)}

if __name__ == "__main__":
    out = {}
    for spec in sys.argv[2:]:
        clip, bd, ad, ref = spec.split(":", 3)
        P = json.load(open(f"{ad}/polylines/{clip}_polylines.json")); n = len(P["frames"])
        v, y = ref_series(ref, n)
        res = {"before": score(run_series(bd, clip, n), v, y), "signfix": score(run_series(bd, clip, n, neg=True), v, y),
               "after": score(run_series(ad, clip, n), v, y)}
        out[clip] = res
        for k, s in res.items():
            print(f"{clip:30s} {k:8s} speed {s['speed']:6.2f} vs {s['speed_ref']:6.2f} ({s['speed_err_pct']:+5.1f}%)  yaw corr {s['yaw_corr']:+.2f} "
                  f"mae {s['yaw_mae_dps']:5.2f} drift {s['heading_drift_deg']:+6.1f}  path end {s['path_end_err_m']:5.2f} m (lat {s['path_lat_end_err_m']:5.2f}) rms {s['path_rms_err_m']:5.2f} / {s['path_len_ref_m']:5.1f} m  yaw jerk {s['yaw_jerk_dps2']:.1f}", flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
