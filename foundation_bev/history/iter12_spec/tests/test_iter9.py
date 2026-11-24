"""iter9 CPU unit tests: kerb-behind-cars rule + hysteresis, centred line smoothing, RHT dot rule.
Run: python3 -m pytest -q tests/test_iter9.py  (or python3 tests/test_iter9.py)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lines_mt as LM
import bev_smooth as BS

Z = LM.ZS
GM = BS.GroundModel(371.8, 923.6, 314.1, 221.3, 1.65, 0.0)  # lane_split K


def _f29():
    # lane_split f29 (GPU iter8): main_R runs along the parked-car bottoms, converging; boxes of the two parked cars
    x = np.where((Z >= 8) & (Z <= 15), 3.36 - (Z - 8) * 0.28, np.nan)
    cars = LM.car_footprints([[350, 234, 452, 372], [307, 237, 397, 334]], GM.lift)
    return x, cars


def test_car_footprints_f29():
    _, cars = _f29()
    assert len(cars) == 2
    c = min(cars, key=lambda c: c["zb"])
    assert 9.5 < c["zb"] < 10.6 and 3.3 < c["xr"] < 4.2, c


def test_kerb_outward_only_and_extent():
    x, cars = _f29()
    xk, app = LM.kerb_behind_cars(x, 1, cars, ref_slope=0.0)
    assert app
    f = np.isfinite(x)
    assert np.array_equal(np.isfinite(xk), f)                 # presence / z-extent unchanged
    assert np.all(xk[f] >= x[f] - 1e-9)                       # outward only (right edge: x grows)
    assert np.all(xk[f] - x[f] <= LM.KERB_MAX_PUSH_M + 1e-9)  # capped


def test_kerb_not_bounded_no_change():
    x = np.where((Z >= 8) & (Z <= 30), 5.0, np.nan)           # edge 5 m right, cars nowhere near
    xk, app = LM.kerb_behind_cars(x, 1, [{"zb": 12.0, "xl": -1.0, "xr": 1.2, "xa": -0.7, "xb": 0.9}], 0.0)
    assert not app and np.allclose(xk[np.isfinite(x)], 5.0)


def test_kerb_skips_traffic_queue():
    # munich f692: a long lane line 1.7 m right of the ego, queue car right of it -> traffic, no kerb push
    x = np.where((Z >= 8) & (Z <= 25), 2.4 + 0.04 * (Z - 8), np.nan)
    car = {"zb": 9.1, "xl": 1.4, "xr": 4.69, "xa": 1.9, "xb": 4.2}
    car["xa"], car["xb"] = 2.2 * 9.1 / 9.1, 2.6
    lane = {"id": 1, "xg": np.where((Z >= 8) & (Z <= 34), 1.7, np.nan)}
    _, app0 = LM.kerb_behind_cars(x, 1, [car], 0.0)
    _, app1 = LM.kerb_behind_cars(x, 1, [car], 0.0, lanes=[lane])
    assert app0 and not app1


def test_kerb_state_hysteresis():
    x, cars = _f29()
    ks = LM.KerbState()
    apps = [ks.step("main_R", x, 1, cars, 0.0, ds=0.0)[1] for _ in range(LM.KERB_ENTER + 2)]
    assert apps[:LM.KERB_ENTER - 1] == [False] * (LM.KERB_ENTER - 1) and all(apps[LM.KERB_ENTER - 1:])
    # a 2-frame burst never applies and is not held
    ks = LM.KerbState(); none = []
    a = [ks.step("main_R", x, 1, cars, 0.0)[1], ks.step("main_R", x, 1, cars, 0.0)[1], ks.step("main_R", x, 1, [], 0.0)[1]]
    assert a == [False, False, False] and "main_R" not in ks.s


def _static_frames(n, ds, slope=0.0, kind="lanes"):
    s = np.arange(n) * ds
    fr = []
    for k in range(n):
        zw = Z + s[k]  # world z of each ego-frame sample
        x = 1.8 + slope * zw
        rec = {"id": 7, "xg": np.where(Z <= 30, x, np.nan)}
        if kind == "edges":
            rec["role"] = "main_R"
        fr.append([rec])
    return fr, s


def test_centre_smooth_static_exact():
    for kind in ("lanes", "edges"):
        fr, s = _static_frames(12, 1.3, slope=0.05, kind=kind)
        out = LM.centre_smooth_lines(fr, s, 10.0, half_s=0.3, kind=kind)
        for a, b in zip(fr, out):
            f = np.isfinite(a[0]["xg"])
            assert np.array_equal(f, np.isfinite(b[0]["xg"]))
            assert np.allclose(a[0]["xg"][f], b[0]["xg"][f], atol=1e-6)


def test_centre_smooth_symmetric_linear_trend():
    # a line drifting linearly in time stays exactly linear (symmetric window, incl. track birth / end)
    n = 15; s = np.zeros(n)
    fr = [[{"id": 3, "xg": np.where(Z <= 20, 1.0 + 0.1 * k, np.nan)}] for k in range(n)]
    out = LM.centre_smooth_lines(fr, s, 10.0, half_s=0.3)
    v = np.array([o[0]["xg"][5] for o in out])
    assert np.allclose(np.diff(v, 2), 0.0, atol=1e-9)


def test_centre_smooth_reduces_noise():
    rng = np.random.default_rng(0); n = 40; s = np.zeros(n)
    fr = [[{"id": 3, "xg": np.where(Z <= 20, 1.0 + rng.normal(0, 0.1), np.nan)}] for _ in range(n)]
    out = LM.centre_smooth_lines(fr, s, 10.0, half_s=0.3)
    a = np.array([f[0]["xg"][5] for f in fr]); b = np.array([o[0]["xg"][5] for o in out])
    assert np.std(np.diff(b, 2)) < 0.5 * np.std(np.diff(a, 2))


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print(k, "ok")
