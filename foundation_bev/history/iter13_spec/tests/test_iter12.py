"""iter12 unit tests: random-walk ('hold') edge backfill (GPU iter11 city / late edge wjit), backfill source that covers
z 10 m + near fill + diagonal-copy gate (munich f0, a2d2a f0), the 1.0 s edge window, the lane-anchored coast
(munich R f561-577, f165-166) and its guards, and the unchanged iter9-11 kerb traffic guard."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lines_mt as LM
Z = LM.ZS
i10 = int(np.argmin(np.abs(Z - 10.0)))


def _line(x0, slope=0.0, z0=6.0, z1=40.0):
    return np.where((Z >= z0) & (Z <= z1), x0 + slope * (Z - 10.0), np.nan)


def _rec(x, role="main_R", side=1, **kw):
    return dict({"id": 4, "side": side, "role": role, "xg": x, "hits": 5, "held": False}, **kw)


def test_defaults():
    assert LM.EDGE_BACKFILL_MODE == "hold" and LM.EDGE_BACKFILL_COVER and abs(LM.EDGE_BACKFILL_S - 1.0) < 1e-9
    assert abs(LM.LANE_BACKFILL_S - 0.6) < 1e-9 and abs(LM.EDGE_ANCHOR_S - 3.0) < 1e-9 and LM.KERB_TRAFFIC == "outer"


def _moving_edge(n=20, k0=3, v=-0.18):
    # city main_L: the live edge moves 0.18 m / frame (from the parked-car bottoms out to the kerb)
    fr = [[] for _ in range(n)]
    for k in range(k0, n):
        fr[k].append(_rec(_line(-6.1 + v * (k - k0), 0.0, 8.0, 40.0), role="main_L", side=-1))
    return fr, np.zeros(n)  # ego standing: no travel shift, only the offset model acts


def test_edge_backfill_hold_vs_cv():
    fr, s = _moving_edge()
    old = LM.EDGE_BACKFILL_MODE
    try:
        LM.EDGE_BACKFILL_MODE = "hold"
        h = LM.edge_backfill(fr, s, 10.0)
        LM.EDGE_BACKFILL_MODE = "cv"
        c = LM.edge_backfill(fr, s, 10.0)
    finally:
        LM.EDGE_BACKFILL_MODE = old
    for k in range(3):
        assert h[k][0]["backfill"] and abs(h[k][0]["xg"][i10] - (-6.1)) < 1e-6        # first live offset
        assert c[k][0]["xg"][i10] > -6.1 + 0.1                                         # cv continues the ramp inward


def test_edge_backfill_source_covers_zref_and_near_fill():
    n = 30; s = np.arange(n) * 0.5; fr = [[] for _ in range(n)]
    for k in range(2, 8):
        fr[k].append(_rec(_line(5.0, 0.0, 17.0, 40.0)))                 # munich f2-7: starts at z 17 m
    fr[8].append(_rec(_line(5.0, 0.0, 17.0, 40.0), held=True))           # a coasted frame in between
    for k in range(9, n):
        fr[k].append(_rec(_line(5.0, 0.0, 9.5, 40.0)))                  # covers z 10 m
    out = LM.edge_backfill(fr, s, 10.0)
    for k in (0, 1):
        assert out[k][0]["backfill"] and abs(out[k][0]["xg"][i10] - 5.0) < 0.05
    for k in range(2, 8):
        r = out[k][0]
        assert r["backfill"] and abs(r["xg"][i10] - 5.0) < 0.05             # near samples added ...
        assert np.allclose(r["xg"][(Z >= 17.0) & (Z <= 40.0)], 5.0)                          # ... the measured part unchanged


def test_edge_backfill_near_fill_needs_a_join():
    n = 30; s = np.arange(n) * 0.5; fr = [[] for _ in range(n)]
    for k in range(0, 5):
        fr[k].append(_rec(_line(7.0, 0.0, 17.0, 40.0)))                 # 2 m off the source at its near end
    for k in range(5, n):
        fr[k].append(_rec(_line(5.0, 0.0, 9.5, 40.0)))
    out = LM.edge_backfill(fr, s, 10.0)
    assert not out[0][0].get("backfill") and not np.isfinite(out[0][0]["xg"][i10])


def test_edge_backfill_rejects_diagonal_copy():
    n = 30; s = np.arange(n) * 1.0; fr = [[] for _ in range(n)]
    for k in range(4, n):
        fr[k].append(_rec(_line(6.0, 0.5, 9.5, 40.0)))                  # diagonal near part (munich f14 source)
    lanes = [[{"xg": _line(1.8)}] for _ in range(n)]
    out = LM.edge_backfill(fr, s, 10.0, lanes=lanes)
    assert not any(out[k] for k in range(4))


def test_edge_backfill_window_1s():
    n = 30; s = np.arange(n) * 1.0; fr = [[] for _ in range(n)]
    for k in range(8, n):
        fr[k].append(_rec(_line(2.5, 0.0, 6.5, 40.0)))                  # a2d2a main_R: born at f8 (0.8 s)
    out = LM.edge_backfill(fr, s, 10.0)
    assert all(out[k] and out[k][0]["backfill"] for k in range(8))


LANES = [{"id": 1, "side": -1, "xg": _line(-1.8), "hits": 9}, {"id": 2, "side": 1, "xg": _line(1.8), "hits": 9}]


def _sl(fps=10.0):
    return LM.StageLines(fps=fps)


def _main_R(ed):
    r = [e for e in ed if e and e.get("role") == "main_R"]
    return r[0] if r else None


def test_lane_anchor_fills_after_the_hold_expires():
    sl = _sl()
    L = _rec(_line(-6.0, 0.0, 9.5, 40.0), role="main_L", side=-1)
    for _ in range(5):
        sl.post_edges(LANES, [dict(L), _rec(_line(5.0, 0.0, 9.5, 40.0))], [], 1.0)
    for k in range(25):                                                  # 2.5 s without a right edge
        r = _main_R(sl.post_edges(LANES, [dict(L)], [], 1.0)[0])
        assert r is not None and r.get("held") and abs(r["xg"][i10] - 5.0) < 0.05
    assert r.get("anchored")
    for k in range(10):                                                  # > EDGE_ANCHOR_S since the last learn
        r = _main_R(sl.post_edges(LANES, [dict(L)], [], 1.0)[0])
    assert r is None


def test_lane_anchor_follows_the_lane_not_the_travel():
    sl = _sl()
    for _ in range(5):
        sl.post_edges(LANES, [_rec(_line(5.0, 0.0, 9.5, 40.0))], [], 1.0)
    bent = [LANES[0], dict(LANES[1], xg=_line(1.8, 0.05))]                # the road (and its lane) curves right
    r = _main_R(sl.post_edges(bent, [], [], 1.0)[0])
    for _ in range(8):
        r = _main_R(sl.post_edges(bent, [], [], 1.0)[0])
    m = (Z >= 6.0) & (Z <= 30.0)
    assert r.get("anchored") and np.allclose(r["xg"][m] - bent[1]["xg"][m], 3.2, atol=1e-6)


def test_lane_anchor_not_learned_from_a_diagonal_edge():
    sl = _sl()
    for _ in range(5):
        sl.post_edges(LANES, [_rec(_line(4.0, 0.66, 9.5, 25.0))], [], 1.0)   # munich f542-545 forecourt hook
    for _ in range(8):
        r = _main_R(sl.post_edges(LANES, [], [], 1.0)[0])
    assert r is None or not r.get("anchored")


def test_lane_anchor_falls_back_to_the_opposite_edge():
    sl = _sl()
    L = _rec(_line(-6.0, 0.0, 9.5, 40.0), role="main_L", side=-1)
    for _ in range(5):
        sl.post_edges(LANES, [dict(L), _rec(_line(5.0, 0.0, 9.5, 40.0))], [], 1.0)
    for _ in range(8):
        r = _main_R(sl.post_edges([LANES[0]], [dict(L)], [], 1.0)[0])     # no right lane: road width from main_L
    assert r.get("anchored") and abs(r["xg"][i10] - 5.0) < 0.05


def test_lane_anchor_keeps_a_held_kerb_line_and_live_edges():
    a = LM.LaneAnchor(fps=10.0); a.s["main_R"] = {"lane": {"off": 3.2, "age": 0}}
    h = _rec(_line(6.0), held=True, kerb_occ=True)
    assert a.fill("main_R", h, 1, LANES) is h                           # iter9 kerb hold reaching z 10 stays
    live = _rec(_line(4.0))
    assert a.fill("main_R", live, 1, LANES) is live
    far = _rec(_line(6.0, 0.0, 22.0, 40.0), held=True, kerb_occ=True, hold_age=3)  # munich f165: EdgeHold kerb from z 22 m
    assert a.fill("main_R", far, 1, LANES).get("anchored")


def test_lane_anchor_keeps_a_tracker_coasted_edge():
    """a record the BEV edge tracker coasts (held, no EdgeHold hold_age) is treated as live: never replaced."""
    a = LM.LaneAnchor(fps=10.0); a.s["main_R"] = {"lane": {"off": 3.2, "age": 0}}
    tc = _rec(_line(5.6), held=True)
    assert a.fill("main_R", tc, 1, LANES) is tc
    hc = _rec(_line(5.6), held=True, hold_age=2)                         # EdgeHold travel coast -> anchored
    o = a.fill("main_R", hc, 1, LANES)
    assert o.get("anchored") and abs(o["xg"][i10] - 5.0) < 1e-6


def test_lane_anchor_extends_a_live_edge_that_misses_zref_only_on_a_join():
    a = LM.LaneAnchor(fps=10.0); a.s["main_R"] = {"lane": {"off": 3.2, "age": 0}}
    ok = _rec(_line(5.1, 0.0, 12.5, 20.0))
    o = a.fill("main_R", ok, 1, LANES)
    assert o.get("anchored") and abs(o["xg"][i10] - 5.1) < 1e-6 and not o.get("held")
    hook = _rec(_line(6.8, 0.0, 12.5, 20.0))                              # munich f572-575: 2 m off the anchor
    assert a.fill("main_R", hook, 1, LANES) is hook


def test_anchor_off_restores_iter11():
    old = LM.EDGE_ANCHOR_S
    try:
        LM.EDGE_ANCHOR_S = 0.0
        sl = _sl()
        for _ in range(5):
            sl.post_edges(LANES, [_rec(_line(5.0, 0.0, 9.5, 40.0))], [], 1.0)
        for _ in range(8):
            r = _main_R(sl.post_edges(LANES, [], [], 1.0)[0])
        assert r is None                                                 # 0.5 s travel hold only
    finally:
        LM.EDGE_ANCHOR_S = old


def test_kerb_traffic_guard_unchanged():
    # a long lane line inside the car's outer flank still marks traffic (munich f692 / f760 queues)
    cars = [{"zb": 12.0, "xl": 2.5, "xr": 4.5, "xa": 2.8, "xb": 4.2}]
    x = _line(3.5, 0.0, 6.0, 30.0)
    lanes = [{"xg": _line(1.8)}, {"xg": _line(3.2)}]
    assert not LM.kerb_behind_cars(x, 1, cars, 0.0, lanes=lanes)[1]
