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
