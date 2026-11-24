"""iter10 unit tests: edge gap bridging before trim (munich R), ego lane-pair width smoothing, clip-start lane
backfill, kerb blend in / out."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lines_mt as LM
Z = LM.ZS
i10 = int(np.argmin(np.abs(Z - 10.0)))


def _edge(x0=5.6, slope=0.05, z0=9.5, z1=44.5):
    return np.where((Z >= z0) & (Z <= z1), x0 + slope * (Z - 10.0), np.nan)


def test_bridge_fills_short_interior_gap_only():
    x = _edge(); x[(Z > 10.6) & (Z < 11.4)] = np.nan; x[(Z > 15.1) & (Z < 30.4)] = np.nan  # 1 m and 15.5 m holes
    b = LM.bridge_gaps(x)
    assert np.isfinite(b[(Z >= 9.5) & (Z <= 15.0)]).all()          # 1 m hole bridged
    assert not np.isfinite(b[(Z > 15.1) & (Z < 30.4)]).any()        # 15.5 m hole (> 5 m) kept
    assert not np.isfinite(b[Z < 9.5]).any()                        # never extrapolated
    assert np.allclose(b[np.isfinite(x)], x[np.isfinite(x)])        # measured samples untouched


def test_bridge_rejects_steep_bridge():
    x = _edge(slope=0.0); x[(Z > 12.1) & (Z < 14.9)] = np.nan; x[Z >= 15.0] += 2.5  # 3 m hole with a 2.5 m jump
    b = LM.bridge_gaps(x)
    assert not np.isfinite(b[(Z > 12.1) & (Z < 14.9)]).any()


def test_trim_keeps_near_after_bridge_munich_f258():
    # munich f258: 9.5-10.5 + hole at 11.0 + 11.5-15.0 + 15.5 m hole: iter9 kept 11.5-15 (z = 10 lost)
    x = _edge(5.71, 0.2); x[(Z > 10.6) & (Z < 11.4)] = np.nan; x[(Z > 15.1) & (Z < 30.4)] = np.nan
    t9 = None
    g = LM.EDGE_GAP_M
    try:
        LM.EDGE_GAP_M = 0.0
        t9 = LM.trim_edge(x, 1, [])
    finally:
        LM.EDGE_GAP_M = g
    t = LM.trim_edge(x, 1, [])
    assert not np.isfinite(t9[i10]) and np.isfinite(t[i10])


def test_bridge_does_not_readmit_flare_or_crossing():
    # diverging 55 deg near ego (forecourt flare, munich f240) with a hole: still trimmed near
    x = np.where(Z <= 20, 4.0 + np.tan(np.radians(55)) * (Z - 9.5), 4.0 + np.tan(np.radians(55)) * 10.5)
    x[(Z > 12.1) & (Z < 13.9)] = np.nan
    t = LM.trim_edge(x, 1, [])
    assert t is None or not np.isfinite(t[i10])
    # an edge that crosses a lane from outside stays cut from the crossing on, hole or not
    lane = {"xg": np.where(Z <= 40, 1.8, np.nan)}
    xc = np.where(Z <= 40, 3.0 - 0.15 * (Z - 9.5), np.nan); xc[(Z > 14.1) & (Z < 15.9)] = np.nan
    tc = LM.trim_edge(xc, 1, [lane])
    assert tc is None or not np.isfinite(tc[Z >= 22.0]).any()      # > 0.5 m inside the lane from z ~ 21


def _pair_frames(n=40, wob=0.3, period=12):
    fr = []
    for k in range(n):
        xl = np.where(Z <= 40, -1.8, np.nan)
        xr = np.where(Z <= 40, 1.8 + wob * np.sin(2 * np.pi * k / period), np.nan)
        fr.append([{"id": 1, "xg": xl}, {"id": 2, "xg": xr}])
    return fr, np.arange(n) * 2.5


def _wjit(fr):
    w = [r[1]["xg"][i10] - r[0]["xg"][i10] for r in fr]
    return float(np.median(np.abs(np.diff(w)) / 2))


def test_pair_width_smooth_reduces_wobble_on_wobbling_line():
    fr, s = _pair_frames()
    out = LM.pair_width_smooth(fr, s, 10.0)
    assert _wjit(out) < 0.5 * _wjit(fr)
    dl = max(abs(o[0]["xg"][i10] - f[0]["xg"][i10]) for o, f in zip(out, fr))
    dr = max(abs(o[1]["xg"][i10] - f[1]["xg"][i10]) for o, f in zip(out, fr))
    assert dl < 0.05 < dr                                         # the wobbling (right) line absorbs it
    assert all(np.array_equal(np.isfinite(o[i]["xg"]), np.isfinite(f[i]["xg"])) for o, f in zip(out, fr) for i in (0, 1))


def test_pair_width_smooth_keeps_real_width_change_at_pair_switch():
    fr, s = _pair_frames(wob=0.0)
    for k in range(20, 40):  # new right line (id 3) 3.5 m further: a different pair, not smoothed across
        fr[k][1] = {"id": 3, "xg": np.where(Z <= 40, 5.3, np.nan)}
    out = LM.pair_width_smooth(fr, s, 10.0)
    assert abs(out[19][1]["xg"][i10] - 1.8) < 1e-6 and abs(out[20][1]["xg"][i10] - 5.3) < 1e-6


def test_lane_backfill_clip_start_only():
    n = 30; s = np.arange(n) * 2.5; fr = [[] for _ in range(n)]
    for k in range(2, n):
        fr[k].append({"id": 5, "xg": np.where((Z >= 9.5) & (Z <= 40), -1.9 + 0.01 * (Z - 10), np.nan), "hits": 4})
    for k in range(15, n):  # a mid-clip birth is not backfilled
        fr[k].append({"id": 9, "xg": np.where((Z >= 9.5) & (Z <= 40), 1.7, np.nan), "hits": 4})
    out = LM.lane_backfill(fr, s, 10.0)
    for k in (0, 1):
        assert len(out[k]) == 1 and out[k][0]["backfill"] and np.isfinite(out[k][0]["xg"][i10])
        assert abs(out[k][0]["xg"][i10] - (-1.9)) < 0.1
    assert len(out[14]) == 1 and not any(r.get("backfill") for r in out[14])


def test_lane_backfill_no_duplicate():
    n = 10; s = np.arange(n) * 2.5; fr = [[] for _ in range(n)]
    for k in range(n):
        if k >= 2:
            fr[k].append({"id": 5, "xg": np.where((Z >= 9.5) & (Z <= 40), -1.9, np.nan)})
    fr[0].append({"id": 4, "xg": np.where((Z >= 9.5) & (Z <= 40), -1.95, np.nan)})  # already a line there
    out = LM.lane_backfill(fr, s, 10.0)
    assert len(out[0]) == 1 and out[0][0]["id"] == 4                 # no duplicate next to the existing line
    assert len(out[1]) == 1 and out[1][0].get("backfill")


def test_kerb_ramp_in_and_out():
    from test_iter9 import _f29
    x, cars = _f29()
    ks = LM.KerbState(fps=10.0)
    assert ks.ramp_n == 3
    pushes = []
    for k in range(LM.KERB_ENTER + 4):
        xo, app = ks.step("main_R", x, 1, cars, 0.0)
        pushes.append(float(np.nanmax(xo - x)) if app else 0.0)
    full = pushes[-1]
    assert full > 0.2
    first = pushes[LM.KERB_ENTER - 1]
    assert 0.2 * full < first < 0.5 * full                         # 1/3 of the push on the entry frame
    # ramp out: cars gone -> the hold, then fades instead of snapping back
    outs = []
    for k in range(LM.KERB_HOLD + 4):
        xo, app = ks.step("main_R", x, 1, [], 0.0)
        outs.append(float(np.nanmax(xo - x)) if app else 0.0)
    nz = [p for p in outs if p > 0]
    assert nz and outs[-1] == 0.0 and any(0 < p < 0.9 * full for p in outs)


def test_kerb_ramp_off_is_iter9():
    from test_iter9 import _f29
    x, cars = _f29()
    old = LM.KERB_RAMP_S
    try:
        LM.KERB_RAMP_S = 0.0
        ks = LM.KerbState(fps=10.0)
        assert ks.ramp_n == 1
        apps = [ks.step("main_R", x, 1, cars, 0.0)[1] for _ in range(LM.KERB_ENTER + 1)]
        assert apps[LM.KERB_ENTER - 1]
    finally:
        LM.KERB_RAMP_S = old
