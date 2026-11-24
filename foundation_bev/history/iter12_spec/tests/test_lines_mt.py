import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import lines_mt as LM, bev_ipm as IP
ZS = LM.ZS


def line(x0, slope=0.0, z0=5.0, z1=45.0, z_div=None, div=0.0):
    x = x0 + slope * (ZS - z0)
    if z_div is not None:
        x = x + np.where(ZS > z_div, div * (ZS - z_div), 0.0)
    x[(ZS < z0) | (ZS > z1)] = np.nan
    return x

# 1) diverging exit line: a through line stays, the exit line is born as a child (fork) and is not averaged
tr = LM.MultiLineTracker("lane")
for f in range(12):
    obs = [line(-1.8), line(1.8)]
    if f >= 4:
        obs.append(line(1.8, z0=15.0, z_div=15.0, div=0.15))      # diverges right from z = 15 m
    live = tr.step(f, obs)
ids = {r["id"]: r for r in live}
assert len(live) == 3, len(live)
child = [r for r in live if r["parent"] is not None]
assert len(child) == 1 and abs(child[0]["fork_z"] - 15.0) <= 1.0, child
thru = [r for r in live if r["id"] == child[0]["parent"]][0]
k = np.searchsorted(ZS, 35.0)
assert abs(thru["x"][k] - 1.8) < 0.1 and abs(child[0]["x"][k] - (1.8 + 0.15 * 20)) < 0.2  # neither averaged
assert any(e["type"] == "fork" for e in tr.events)

# 2) gate is relative: a 0.8 m lateral step (ego lane change) on a quiet line is a new line, a steady
#    0.15 m/frame drift is followed by the offset-rate model with the same id
tr = LM.MultiLineTracker("lane")
for f in range(10):
    live = tr.step(f, [line(1.8 - 0.15 * f)])
assert len(tr.hist) == 1, len(tr.hist)

# 3) birth after n_birth, death after n_death misses, no slot limit (6 lines)
tr = LM.MultiLineTracker("lane", n_birth=3, n_death=4)
for f in range(3):
    live = tr.step(f, [line(x) for x in (-9, -5.5, -1.8, 1.8, 5.5, 9)])
assert len(live) == 6
for f in range(3, 9):
    live = tr.step(f, [])
assert len(live) == 0

# 4) smoothing spline keeps a bend that a single global poly2 would average
z = np.arange(6, 40, 0.5); x = np.where(z < 20, 1.8, 1.8 + 0.2 * (z - 20)) + np.random.default_rng(0).normal(0, 0.03, len(z))
xs = LM.smooth_chain(np.stack([x, z], 1))
k = np.searchsorted(ZS, 38.0)
assert abs(xs[k] - (1.8 + 0.2 * 18)) < 0.25, xs[k]

# 5) drivable area that widens (exit lane) then forks around a grass island -> main + split edges
H, W = IP.bev_shape()
road = np.zeros((H, W), np.uint8)
for zz in np.arange(5, 45, 0.1):
    v = int(IP.xz_to_bev_px(0, zz)[1])
    xr = 2.0 + (0.25 * (zz - 12) if zz > 12 else 0.0)           # right boundary widens from z = 12 m
    road[v, int(IP.xz_to_bev_px(-5.0, zz)[0]):int(IP.xz_to_bev_px(xr, zz)[0])] = 1
    if zz > 28:                                                 # 2 m grass island from z = 28 m
        road[v, int(IP.xz_to_bev_px(2.2, zz)[0]):int(IP.xz_to_bev_px(4.2, zz)[0])] = 0
ed = LM.edge_chains(road > 0, IP.RES)
k20, k35 = np.searchsorted(ZS, 20.0), np.searchsorted(ZS, 35.0)
assert abs(ed["main_R"][k20] - (2.0 + 0.25 * 8)) < 0.3, ed["main_R"][k20]      # widening followed
assert abs(ed["main_R"][k35] - 2.2) < 0.3 and abs(ed["split_in"][k35] - 4.2) < 0.3, (ed["main_R"][k35], ed["split_in"][k35])
assert abs(ed["split_out"][k35] - (2.0 + 0.25 * 23)) < 0.3

# 6) gore: parallel oblique stripes cluster into one polygon; one stray stripe does not
stripes = []
for i in range(4):
    c = np.array([3.0, 18.0 + 2.0 * i]); d = np.array([np.sin(0.8), np.cos(0.8)])
    P = c + np.outer(np.linspace(-1, 1, 20), d)
    stripes.append({"pts": P, "centre": c, "ang": 0.8, "len": 2.0})
g = LM.gore_polygons(stripes)
assert len(g) == 1 and g[0]["n_stripes"] == 4 and g[0]["area"] > 2.0
assert LM.gore_polygons(stripes[:1]) == []
print("ALL ITER4 LINE TESTS PASSED")


def test_stage_lines_and_kernel_hooks():
    """iter4 wiring: StageLines on a synthetic forking road; legacy records draw via xg; BEV-phase speed selection."""
    import bev_smooth as BS, bev_ipm as IP
    import foundation_bev as FB
    hw = IP.bev_shape(); vv, uu = np.mgrid[: hw[0], : hw[1]]
    x, z = IP.bev_px_to_xz(uu, vv)
    road = (np.abs(x) < 5.5 + np.clip(z - 20, 0, None) * 0.25)
    paint = np.zeros(hw, bool)
    for x0 in (-1.75, 1.75):
        paint |= np.abs(x - x0) < 0.08
    valid = np.ones(hw, bool)
    st = LM.StageLines()
    for k in range(6):
        lanes, edges, gores = st.step(k, road, paint, valid, IP.RES)
    assert len(lanes) >= 2 and {e["role"] for e in edges} >= {"main_L", "main_R"}
    for r in lanes + edges:
        pts = BS.sample_curve(r, 0.5, 40.0)
        assert len(pts) > 10 and np.isfinite(pts).all()
    R = [e for e in edges if e["role"] == "main_R"][0]
    xs = BS.sample_curve(R, 0.5, 40.0)
    assert xs[-1, 0] - xs[0, 0] > 2.0, "right edge must follow the widening drivable area"
    cam = {}
    scene = {"speed": np.full(10, 20.0), "valid_frac": 0.0}
    vo = {"speed": np.full(10, 0.4), "speed_median_mps": 0.4, "valid_frac": 0.9, "yaw_rate": np.zeros(10), "yaw": np.zeros(10)}
    ok = FB.select_ego_speed(cam, vo, scene, 22.0, 10, 10.0, bev={"speed_median_mps": 23.6, "valid_frac": 0.7, "n_repeat": 0})
    bad = FB.select_ego_speed(cam, vo, scene, 22.0, 10, 10.0, bev={"speed_median_mps": 3.0, "valid_frac": 0.2, "n_repeat": 0})
    assert ok["source"] == "bev_phase" and bad["source"] == "prior", (ok, bad)


def test_iter5_clean_lanes_and_fork_gore():
    """iter5: clutter filter (short / young / near-horizontal / crossing / per-side cap) and fork gores."""
    Z = LM.ZS
    def rec(i, x, hits=10, parent=None, side=None):
        x = np.asarray(x, float)
        f = np.isfinite(x)
        return {"id": i, "xg": x, "hits": hits, "parent": parent, "fork_z": None, "held": False,
                "side": side if side is not None else (-1 if np.nanmedian(x) < 0 else 1),
                "zmin": float(Z[f].min()), "zmax": float(Z[f].max()), "c": np.zeros(3)}
    good = rec(1, np.full(len(Z), 1.75))
    short = rec(2, np.where(Z < 9, 5.25, np.nan), hits=2)  # iter6: short AND young -> dropped
    young = rec(3, np.full(len(Z), 5.25), hits=2)
    horiz = rec(4, np.where((Z > 10) & (Z < 14), (Z - 10) * 3.0, np.nan))
    cross = rec(5, 1.75 + (Z - 20) * 0.1, hits=5)
    many = [rec(10 + j, np.full(len(Z), 1.75 + 3.5 * j)) for j in range(1, 6)]
    out = LM.clean_lanes([good, short, young, horiz, cross] + many)
    ids = {r["id"] for r in out}
    assert 1 in ids and not ids & {2, 3, 4, 5}, ids
    assert sum(r["side"] == 1 for r in out) <= 4
    # iter6: short-but-old and long-but-young lanes survive
    keep = LM.clean_lanes([rec(30, np.where(Z < 9, -5.25, np.nan), hits=10), rec(31, np.full(len(Z), -1.75), hits=2)])
    assert {r["id"] for r in keep} == {30, 31}
    # fork gore: child diverging from parent with a growing gap
    par = rec(20, np.full(len(Z), 1.75))
    ch = rec(21, np.where(Z >= 15, 1.75 + (Z - 15) * 0.15, np.nan), parent=20)
    g = LM.fork_gores([par, ch], [])
    assert len(g) == 1 and g[0]["src"] == "fork" and g[0]["area"] > 1.0
    assert LM.fork_gores([par, rec(22, np.full(len(Z), 5.25), parent=20)], []) == []  # parallel: no gore



def test_iter6_edge_heading_and_persistent_divergence_gore():
    Z = LM.ZS
    assert LM.heading_ok(np.full(len(Z), 5.0))
    assert not LM.heading_ok(-6.0 + (Z - 5.0) * 0.5)  # ~27 deg diagonal across lanes
    lane_ = {"xg": np.full(len(Z), -1.75)}
    assert LM.crosses_lane(-3.0 + (Z - 5.0) * 0.12, [lane_]) and not LM.crosses_lane(np.full(len(Z), -5.0), [lane_])
    def rec(i, x, side, role=None, hits=10):
        x = np.asarray(x, float); f = np.isfinite(x)
        return {"id": i, "xg": x, "hits": hits, "parent": None, "fork_z": None, "held": False, "side": side,
                "zmin": float(Z[f].min()), "zmax": float(Z[f].max()), "c": np.zeros(3), "role": role}
    lane = rec(1, np.full(len(Z), 1.75), 1)
    dg = LM.DivergenceGore()
    fired = []
    for k in range(10):  # right edge moves out 0.3 m per frame (exit lane opening as the ego approaches)
        e = rec(2, np.full(len(Z), 3.5 + 0.3 * k), 1, role="main_R")
        fired.append(bool(dg.step([lane], [e])))
    assert not any(fired[:4]) and any(fired[4:]), fired
    dg3 = LM.DivergenceGore()  # a 1.5 m jump (lane change / edge flip) is not a gore
    assert not any(dg3.step([lane], [rec(2, np.full(len(Z), x), 1, role="main_R")]) for x in [3.5] * 4 + [5.0] * 6)
    dg2 = LM.DivergenceGore()
    jit = [3.5, 4.6, 3.4, 4.7, 3.5, 4.6]  # flicker, not growth -> no gore
    assert not any(dg2.step([lane], [rec(2, np.full(len(Z), x), 1, role="main_R")]) for x in jit)


def _r(x, side=None, role=None, hits=10, i=0):
    Z = LM.ZS
    x = np.asarray(x, float); f = np.isfinite(x)
    return {"id": i, "xg": x, "hits": hits, "parent": None, "fork_z": None, "held": False,
            "side": side if side is not None else (-1 if np.nanmedian(x) < 0 else 1),
            "zmin": float(Z[f].min()), "zmax": float(Z[f].max()), "c": np.zeros(3), "role": role}


def test_iter7_trim_edge_keeps_good_part():
    Z = LM.ZS
    lane = _r(np.full(len(Z), -1.75))
    straight = np.full(len(Z), -5.0)
    assert np.isfinite(LM.trim_edge(straight, -1, [lane])).sum() == np.isfinite(straight).sum()
    # lane_split f0015: left edge starts outside the lane (x=-5 at z=5) and runs diagonally inward across it
    # (crossing at z~16): only the part on the correct side of the lane survives
    diag = -5.0 + (Z - 5.0) * 0.3
    t = LM.trim_edge(diag, -1, [lane])
    zk = LM.ZS[np.isfinite(t)]
    assert t is not None and zk.min() <= 5.5 and zk.max() <= 18.5, (zk.min(), zk.max())
    assert np.all(t[np.isfinite(t)] <= -1.75 + 0.5)
    # paint lying BEYOND the edge (parking line at x=-6.5) must not trim it
    assert np.isfinite(LM.trim_edge(straight, -1, [_r(np.full(len(Z), -6.5))])).sum() == np.isfinite(straight).sum()
    # iter8: a near-ego bend toward the ego is cut where it enters the ego corridor (|x| < 1.2 m), the rest is kept
    bent = np.where(Z < 15, -5.0 + (15 - Z) * np.tan(np.radians(30)), -5.0)
    t2 = LM.trim_edge(bent, -1, [])
    kept = np.isfinite(t2)
    assert t2 is not None and np.all(t2[kept] <= -1.2) and kept[LM.ZS >= 10].all()


def test_iter7_edge_hold_propagates_and_fills():
    Z = LM.ZS
    h = LM.EdgeHold(max_age=5)
    e = _r(np.full(len(Z), -5.0), role="main_L")
    assert h.step("main_L", e, 1.0)["xg"] is not None
    outs = [h.step("main_L", None, 1.0) for _ in range(7)]
    assert all(o is not None and o["held"] for o in outs[:5]) and outs[5] is None and outs[6] is None
    assert outs[0]["hold_conf"] > outs[4]["hold_conf"]
    h2 = LM.EdgeHold()
    h2.step("main_R", _r(np.full(len(Z), 5.0), role="main_R"), 0.5)
    trimmed = np.where(Z >= 15, 5.0, np.nan)            # near part trimmed this frame
    o = h2.step("main_R", _r(trimmed, role="main_R"), 0.5)
    assert o["zmin"] < 6.0 and o.get("hold_filled", 0) > 0


def test_iter7_lane_duplicating_edge_suppressed_but_nearest_kept():
    Z = LM.ZS
    near = _r(np.full(len(Z), 1.7), i=1)
    dup = _r(np.full(len(Z), 5.2), i=2)                 # 0.3 m from the right edge (exit_approach f0020)
    edge = _r(np.full(len(Z), 5.5), role="main_R")
    ids = {r["id"] for r in LM.suppress_edge_duplicates([near, dup], [edge])}
    assert ids == {1}
    ids = {r["id"] for r in LM.suppress_edge_duplicates([dup], [edge])}  # nearest on its side -> kept
    assert ids == {2}


# ---------------------------------------------------------------- iter8


def test_iter8_crossing_invalidates_beyond_and_keeps_near_part():
    """lane_split f23 (iter7 GPU): right edge 4.4 m at z 8 -> 0 m at z 30, crossing the lane at 4.2 m near z 9.
    iter7 kept the far diagonal (longest run); iter8 keeps the near part and nothing beyond the crossing."""
    Z = LM.ZS
    lane = _r(np.where(Z <= 20, 4.2, np.nan))
    edge = np.where(Z >= 8, 4.4 - (Z - 8) * 0.2, np.nan)
    t = LM.trim_edge(edge, 1, [lane], min_len_m=1.0)
    kept = LM.ZS[np.isfinite(t)]
    assert kept.min() == 8.0 and kept.max() < 12.0  # crossing (> 0.5 m inside the lane) starts at z 12
    import importlib.util, os
    p7 = os.path.join(os.path.dirname(__file__), "..", "..", "iter7_spec", "src", "lines_mt.py")
    if os.path.exists(p7):  # the iter7 rule kept the far diagonal instead
        sp = importlib.util.spec_from_file_location("lm7", p7); lm7 = importlib.util.module_from_spec(sp); sp.loader.exec_module(lm7)
        k7 = LM.ZS[np.isfinite(lm7.trim_edge(edge, 1, [lane]))]
        assert k7.min() > 12.0


def test_iter8_ego_corridor_and_direction_aware_heading():
    Z = LM.ZS
    # converging 35 deg (toward the lanes with distance) near ego: trimmed (limit 30)
    conv = np.where(Z <= 20, 5.0 - (Z - 5.0) * np.tan(np.radians(35)), np.nan) + 0.0
    conv = np.where(np.isfinite(conv), conv + 0.0, np.nan)
    t = LM.trim_edge(np.where(Z <= 20, 8.0 - (Z - 5.0) * np.tan(np.radians(35)), np.nan), 1, [])
    assert t is None or np.isfinite(t).sum() <= 2
    # diverging 40 deg (munich-R corner flare, away from the road): kept (limit 45) and flagged as a flare
    div40 = np.where(Z <= 20, 3.0 + (Z - 5.0) * np.tan(np.radians(40)), np.nan)
    t40 = LM.trim_edge(div40, 1, [])
    assert t40 is not None and np.isfinite(t40).sum() == np.isfinite(div40).sum()
    assert LM.is_flared(t40, 1, []) and not LM.is_flared(np.full(len(Z), 5.0), 1, [])
    # diverging 50 deg: trimmed
    div50 = np.where(Z <= 20, 3.0 + (Z - 5.0) * np.tan(np.radians(50)), np.nan)
    assert LM.trim_edge(div50, 1, []) is None
    # an edge inside the ego corridor is never kept
    assert LM.trim_edge(np.full(len(Z), 0.8), 1, []) is None


def test_iter8_hold_is_time_based():
    assert LM.EdgeHold(fps=10.0).max_age == 5
    assert LM.EdgeHold(fps=30.0).max_age == 15  # munich: 0.5 s instead of 0.17 s
    h = LM.EdgeHold(fps=30.0)
    h.step("main_R", _r(np.full(len(LM.ZS), 5.0), role="main_R"), 0.4)
    outs = [h.step("main_R", None, 0.4) for _ in range(16)]
    assert all(o is not None and o["held"] for o in outs[:15]) and outs[15] is None


def test_iter8_fill_may_not_cost_measured_samples():
    """city f23: an old fill that crosses a lane must not remove the current, valid edge."""
    Z = LM.ZS
    lane = _r(np.full(len(Z), -4.2))
    cur = _r(np.where(Z >= 10, -7.4, np.nan), role="main_L")
    filled = dict(cur, xg=np.where(Z >= 10, -7.4, -3.0), hold_filled=10)  # near fill inside the lane at -4.2
    out = LM.retrim_held(filled, cur, -1, [lane])
    assert out is cur


def test_iter8_flared_main_edge_seeds_no_split_fork():
    Z = LM.ZS
    main = dict(_r(np.full(len(Z), 6.0), role="main_R"), side=1)
    split = dict(_r(np.where(Z >= 8, 6.0 + (Z - 8) * 0.25, np.nan), role="split_in"), side=1)
    assert LM.fork_gores([], [main, split], lane_edge=False)
    main["flare"] = True
    assert not LM.fork_gores([], [main, split], lane_edge=False)
