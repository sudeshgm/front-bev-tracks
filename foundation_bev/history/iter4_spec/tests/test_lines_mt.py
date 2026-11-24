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
