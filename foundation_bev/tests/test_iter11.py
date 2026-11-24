"""iter11 unit tests: kerb-blend restart of the widen-gore history (GPU iter10 munich f881-885), diagonal fragment
veto (munich R f520-526 / f557-560), fragment extension, 'kf' pair width (slow random walk, RTS), edge backfill, the
0.6 s clip-start window, and StageLines.post_edges (the replayed step tail)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lines_mt as LM
Z = LM.ZS
i10 = int(np.argmin(np.abs(Z - 10.0)))


def _line(x0, slope=0.0, z0=6.0, z1=40.0):
    return np.where((Z >= z0) & (Z <= z1), x0 + slope * (Z - 10.0), np.nan)


def _run_div(gaps, alphas, reset):
    old = LM.GORE_KERB_RESET
    try:
        LM.GORE_KERB_RESET = reset
        dg = LM.DivergenceGore(); fired = []
        lane = {"id": 1, "side": 1, "xg": _line(1.8)}
        for k, (g, a) in enumerate(zip(gaps, alphas)):
            e = {"id": 7, "side": 1, "role": "main_R", "xg": _line(1.8 + g)}
            if a > 0:
                e.update(kerb_occ=True, kerb_alpha=a)
            if any(x["src"] == "widen" for x in dg.step([lane], [e])):
                fired.append(k)
        return fired
    finally:
        LM.GORE_KERB_RESET = old


def test_kerb_blend_does_not_fire_widen_gore():
    # munich f866-885: kerb push blended in over ramp_n frames, +0.25-0.37 m / frame (each step < max_step 0.4)
    gaps = [3.4] * 4 + [3.4 + 0.3 * i for i in range(1, 6)] + [4.9] * 8
    alphas = [0.0] * 4 + [min(1.0, i / 5.0) for i in range(1, 6)] + [1.0] * 8
    assert _run_div(gaps, alphas, reset=False)                        # iter10 behaviour: a false widen gore
    assert not _run_div(gaps, alphas, reset=True)                     # iter11: the history restarts


def test_widen_gore_still_fires_on_a_real_opening_exit_lane():
    # exit_approach f8-f26: the nearest-line -> edge gap opens 2.2 -> 4.0 m without any kerb
    gaps = [2.2] * 3 + [2.2 + 0.2 * i for i in range(1, 12)]
    assert _run_div(gaps, [0.0] * len(gaps), reset=True)


def test_diag_fragment_veto():
    lanes = [{"xg": _line(-1.8)}, {"xg": _line(1.8)}]
    frag = {"xg": _line(4.0, 0.37, 20.5, 24.5)}                       # munich f520: slope 0.37, z 20.5-24.5
    assert LM.is_diag_fragment(frag, lanes)
    assert not LM.is_diag_fragment({"xg": _line(4.0, 0.02, 20.5, 24.5)}, lanes)   # along the road
    assert not LM.is_diag_fragment({"xg": _line(4.0, 0.37, 9.5, 14.0)}, lanes)    # reaches z 10
    assert not LM.is_diag_fragment({"xg": _line(4.0, 0.37, 12.0, 30.0)}, lanes)   # not short
    assert not LM.is_diag_fragment(dict(frag, held=True, hold_age=2), lanes)   # EdgeHold coast: never vetoed
    assert LM.is_diag_fragment(dict(frag, held=True), lanes)   # iter13: a tracker-coasted fragment is vetoed too
    old = LM.EDGE_FRAG_COAST
    try:
        LM.EDGE_FRAG_COAST = False
        assert not LM.is_diag_fragment(dict(frag, held=True), lanes)                 # iter11-12 behaviour
    finally:
        LM.EDGE_FRAG_COAST = old


def test_post_edges_hold_coasts_over_a_vetoed_fragment():
    sl = LM.StageLines(fps=30.0)
    lanes = [{"id": 1, "side": -1, "xg": _line(-1.8), "hits": 9}, {"id": 2, "side": 1, "xg": _line(1.8), "hits": 9}]
    good = {"id": 7, "side": 1, "role": "main_R", "xg": _line(4.3, 0.0, 9.5, 40.0), "hits": 9, "held": False}
    frag = {"id": 7, "side": 1, "role": "main_R", "xg": _line(4.0, 0.37, 20.5, 24.5), "hits": 9, "held": False}
    for _ in range(5):
        sl.post_edges(lanes, [dict(good)], [], 0.4)
    ed, _ = sl.post_edges(lanes, [dict(frag)], [], 0.4)
    r = [e for e in ed if e and e.get("role") == "main_R"]
    assert r and r[0].get("held") and np.isfinite(r[0]["xg"][i10]) and abs(r[0]["xg"][i10] - 4.3) < 0.1


def test_frag_extend_along_heading():
    lanes = [{"xg": _line(-1.8)}, {"xg": _line(1.8)}]
    h = {"xg": _line(4.0, 0.0, 15.0, 19.5), "held": False}
    o = LM.frag_extend(h, 1, lanes)
    assert o.get("frag_override") and abs(o["xg"][i10] - 4.0) < 1e-6
    d = {"xg": _line(4.0, 0.37, 15.0, 19.5), "held": False}           # diagonal: never extended
    assert LM.frag_extend(d, 1, lanes) is d


def test_kf_rw_smoother_flattens_slow_wobble_and_follows_level():
    t = np.arange(40) / 10.0
    y = 3.6 + 0.3 * np.sin(2 * np.pi * np.arange(40) / 25.0)         # exit: 3.43 <-> 4.01 m over ~25 frames
    w = LM.kf_trend_smooth(t, y)
    assert np.std(np.diff(w)) < 0.5 * np.std(np.diff(y))
    y2 = np.where(np.arange(40) < 20, 3.5, 3.5); y2[5] = np.nan       # gaps are fine
    assert np.isfinite(LM.kf_trend_smooth(t, y2)).all()


def _pair_frames(n=40, wob=0.3, period=25):
    fr = []
    for k in range(n):
        fr.append([{"id": 1, "xg": _line(-1.8)}, {"id": 2, "xg": _line(1.8 + wob * np.sin(2 * np.pi * k / period))}])
    return fr, np.arange(n) * 2.5


def _wjit(fr):
    w = [r[1]["xg"][i10] - r[0]["xg"][i10] for r in fr]
    return float(np.median(np.abs(np.diff(w)) / 2))


def test_pair_kf_beats_tri_on_slow_drift_and_keeps_pair_switch():
    fr, s = _pair_frames()
    kf = LM.pair_width_smooth(fr, s, 10.0, mode="kf")
    tri = LM.pair_width_smooth(fr, s, 10.0, mode="tri")
    assert _wjit(kf) < 0.6 * _wjit(tri)
    fr2, s2 = _pair_frames(wob=0.0)
    for k in range(20, 40):
        fr2[k][1] = {"id": 3, "xg": _line(5.3)}
    out = LM.pair_width_smooth(fr2, s2, 10.0, mode="kf")
    assert abs(out[19][1]["xg"][i10] - 1.8) < 1e-6 and abs(out[20][1]["xg"][i10] - 5.3) < 1e-6


def test_pair_kf_short_run_falls_back_to_tri():
    fr, s = _pair_frames(n=8)                                          # 0.8 s < PAIR_KF_MIN_S
    a = LM.pair_width_smooth(fr, s, 10.0, mode="kf"); b = LM.pair_width_smooth(fr, s, 10.0, mode="tri")
    assert all(np.allclose(np.nan_to_num(x[i]["xg"]), np.nan_to_num(y[i]["xg"])) for x, y in zip(a, b) for i in (0, 1))


def _edge_frames(n=20, k0=3, role="main_R", x=5.0):
    fr = [[] for _ in range(n)]
    for k in range(k0, n):
        fr[k].append({"id": 4, "side": 1, "role": role, "xg": _line(x, 0.0, 9.5, 40.0), "hits": 5, "held": False})
    return fr, np.arange(n) * 2.5


def test_edge_backfill_clip_start():
    fr, s = _edge_frames(k0=4)                                         # final R: confirmed at f4
    out = LM.edge_backfill(fr, s, 10.0)
    for k in range(4):
        assert len(out[k]) == 1 and out[k][0]["backfill"] and out[k][0]["role"] == "main_R"
        assert abs(out[k][0]["xg"][i10] - 5.0) < 0.1
    fr2, s2 = _edge_frames(k0=12)                                      # mid-clip birth (> 0.6 s): untouched
    assert all(not f for f in LM.edge_backfill(fr2, s2, 10.0)[:12])


def test_edge_backfill_needs_confirmation_and_no_duplicate():
    fr, s = _edge_frames(k0=2)
    for k in range(3, 20):
        fr[k] = []                                                     # one-off edge: not confirmed
    assert not LM.edge_backfill(fr, s, 10.0)[0]
    fr, s = _edge_frames(k0=2)
    fr[0].append({"id": 9, "side": 1, "role": "main_R", "xg": _line(6.0, 0.0, 9.5, 40.0), "hits": 5, "held": False})
    out = LM.edge_backfill(fr, s, 10.0)
    assert len(out[0]) == 1 and out[0][0]["id"] == 9 and not out[1]   # present from f0: not a clip-start birth


def test_edge_backfill_respects_lane_crossing():
    fr, s = _edge_frames(k0=2, x=1.0)                                  # an "edge" inside the ego lane
    lanes = [[{"xg": _line(1.8)}] for _ in range(20)]
    out = LM.edge_backfill(fr, s, 10.0, lanes=lanes)
    assert not out[0] or not np.isfinite(out[0][0]["xg"]).any() or out[0][0]["xg"][i10] > 1.8


def test_lane_backfill_window_covers_final_f5():
    n = 30; s = np.arange(n) * 2.5; fr = [[] for _ in range(n)]
    for k in range(5, n):
        fr[k].append({"id": 18, "xg": _line(1.7, 0.0, 9.5, 40.0), "hits": 4})
    out = LM.lane_backfill(fr, s, 10.0)
    assert all(len(out[k]) == 1 and out[k][0]["backfill"] for k in range(5))
