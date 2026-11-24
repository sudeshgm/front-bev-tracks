"""iter5b: lines_mt.fork_gores must work on NumPy 1.x (no np.trapezoid) and on NumPy 2.x (no np.trapz)."""
import importlib, os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _fork_pair():
    import lines_mt as LM
    Z = LM.ZS
    def rec(i, x, parent=None):
        x = np.asarray(x, float); f = np.isfinite(x)
        return {"id": i, "xg": x, "hits": 10, "parent": parent, "fork_z": None, "held": False, "side": 1,
                "zmin": float(Z[f].min()), "zmax": float(Z[f].max()), "c": np.zeros(3)}
    par = rec(20, np.full(len(Z), 1.75))
    ch = rec(21, np.where(Z >= 15, 1.75 + (Z - 15) * 0.15, np.nan), parent=20)
    return LM, par, ch


def _check(LM, par, ch):
    g = LM.fork_gores([par, ch], [])
    assert len(g) == 1 and g[0]["src"] == "fork"
    P = np.asarray(g[0]["poly"])
    x, z = P[:, 0], P[:, 1]
    shoelace = 0.5 * abs(np.dot(x, np.roll(z, 1)) - np.dot(z, np.roll(x, 1)))
    assert g[0]["area"] > 1.0 and abs(g[0]["area"] - shoelace) / shoelace < 0.05, (g[0]["area"], shoelace)


@pytest.mark.parametrize("api", ["numpy1", "numpy2"])
def test_fork_gore_area_both_numpy_apis(monkeypatch, api):
    impl = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    if api == "numpy1":
        monkeypatch.delattr(np, "trapezoid", raising=False)
        monkeypatch.setattr(np, "trapz", impl, raising=False)
    else:
        monkeypatch.delattr(np, "trapz", raising=False)
        monkeypatch.setattr(np, "trapezoid", impl, raising=False)
    sys.modules.pop("lines_mt", None)
    LM, par, ch = _fork_pair()
    assert LM._trapz is impl
    _check(LM, par, ch)
    sys.modules.pop("lines_mt", None)


def test_no_numpy2_only_calls_in_src():
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    bad = ("np.trapezoid(", "np.concat(", "np.unstack(", "np.isdtype(", "np.cumulative_sum(", "np.vecdot(",
           "np.matrix_transpose(", "np.permute_dims(", "np.pow(", "np.acos(", "np.asin(", "np.atan2(", "np.atan(")
    hits = [(f, b) for f in os.listdir(src) if f.endswith(".py") for b in bad if b in open(os.path.join(src, f)).read()]
    assert not hits, hits


def test_iter7_robust_kf_update_limits_outlier_pull():
    import bev_smooth as BS
    kf_args = None
    tr = BS.ObjTracker(10.0, 8.0)
    d = lambda z: {"cls": "car", "score": 0.9, "box": [100, 100, 140, 130], "bev": (-15.0, z), "src": "fuse"}
    for k in range(8):
        tr.step(k, [d(24.0)])
    t = tr.active[0]
    before = float(t["kf"].x[1])
    tr.step(8, [dict(d(24.0), bev=(-15.0, 34.5 - 8.0 * 8 / 10.0 * 0))])  # 10 m depth jump, same box
    after = float(t["kf"].x[1])
    assert abs(after - before) < 3.0, (before, after)



def test_iter8_far_model_only_in_final_pass():
    import bev_smooth as BS
    a = BS._KF(-12.0, 24.0, "car", 0.1); b = BS._KF(-12.0, 24.0, "car", 0.1, final=True)
    for kf in (a, b):
        kf.predict()
        kf.update(-15.0, 30.0, 30.0, "wd3d")  # a2d2a-style range jump along the ray at 30 m
    pull = lambda kf: float(np.hypot(kf.x[0] + 12.0, kf.x[1] - 24.0))
    assert pull(b) < pull(a)                 # the final pass trusts the far range jump less
    c = BS._KF(1.0, 8.0, "car", 0.1); d = BS._KF(1.0, 8.0, "car", 0.1, final=True)
    for kf in (c, d):
        kf.predict(); kf.update(1.2, 8.3, 8.3, "wd3d")
    assert np.allclose(c.x, d.x)             # near tracks (< 18 m, inside 2 sigma): identical
    assert BS.far_weight(10.0) == 0.0 and BS.far_weight(30.0) == 1.0
