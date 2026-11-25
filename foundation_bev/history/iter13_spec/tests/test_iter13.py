"""iter13 unit tests: synthetic main edges never inside parked cars (car_inside / car_clear line, local, drop;
edge_car_clear), LaneAnchor continuity limit, tracker-coast fragment veto (munich R f562-563 / f570-571), the
frame-0 anchored edge backfill, and the low-frame-rate time base (fc1_merge, 4 fps)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lines_mt as LM
Z = LM.ZS
i10 = int(np.argmin(np.abs(Z - 10.0)))


def _line(x0, slope=0.0, z0=6.0, z1=40.0):
    return np.where((Z >= z0) & (Z <= z1), x0 + slope * (Z - 10.0), np.nan)


CAR_R = {"zb": 12.0, "xl": 2.6, "xr": 4.6, "xa": 2.9, "xb": 4.3}  # parked car on the right, inner flank 2.6 m


# --- low-frame-rate time base --------------------------------------------------------------------------------------
def test_fr10_noop_at_10_and_30_fps():
    for n in (3, 4, 5, 6, 10, 12, 15, 40):
        assert LM.fr10(n, 10.0) == n and LM.fr10(n, 30.0) == n and LM.fr10(n, 10.0, 2) == n


def test_fr10_scales_below_10_fps():
    assert LM.fr10(10, 4.0) == 4 and LM.fr10(40, 4.0) == 16 and LM.fr10(12, 4.0, 4) == 5
    assert LM.fr10(3, 4.0, 2) == 2 and LM.fr10(5, 4.0, 2) == 2  # floors
    old = LM.LOWFPS_TIME
    try:
        LM.LOWFPS_TIME = False
        assert LM.fr10(10, 4.0) == 10
    finally:
        LM.LOWFPS_TIME = old


def test_stagelines_counts_unchanged_at_10hz_and_time_based_at_4hz():
    s10, s4 = LM.StageLines(fps=10.0), LM.StageLines(fps=4.0)
    assert s10.div.n_persist == 3 and s10.div.hist == 12 and s10.lane_min_hits == 4 and s10.hold.max_age == 5
    assert LM.StageLines(fps=30.0).hold.max_age == 15
    assert s4.div.n_persist == 2 and s4.div.hist == 5 and s4.lane_min_hits == 2 and s4.hold.max_age == 2


# --- car footprint test and clearance ------------------------------------------------------------------------------
def test_car_inside_hits_only_lines_through_the_footprint():
    assert LM.car_inside(_line(3.4), 1, [CAR_R])            # through the car (2.6 .. 4.6)
    assert not LM.car_inside(_line(5.0), 1, [CAR_R])        # behind it (kerb)
    assert not LM.car_inside(_line(2.0), 1, [CAR_R])        # in front of its inner flank (lane side)
    assert not LM.car_inside(_line(-3.4), -1, [CAR_R])      # other side
    assert not LM.car_inside(_line(3.4, z0=20.0), 1, [CAR_R])  # car z range 11.5 .. 16.5 not covered


def test_car_clear_pushes_out_along_the_road_heading():
    x, st = LM.car_clear(_line(3.4), 1, [CAR_R], 0.0)
    assert st == "pushed" and not LM.car_inside(x, 1, [CAR_R])
    m = np.isfinite(x) & (Z >= 11.5) & (Z <= 16.5)
    assert (x[m] >= CAR_R["xl"] + LM.EDGE_CAR_W + LM.EDGE_CAR_TOL - 1e-6).all()  # cleared the 2 m footprint
    assert (x[np.isfinite(x)] >= _line(3.4)[np.isfinite(x)] - 1e-9).all()        # only outward


def test_car_clear_clear_and_drop():
    x0 = _line(5.2)
    x, st = LM.car_clear(x0, 1, [CAR_R], 0.0)
    assert st == "clear" and np.array_equal(x, x0, equal_nan=True)
    near = dict(CAR_R, zb=8.0)  # cannot leave the car within the cap and < EDGE_CAR_TRIM_MIN m before it
    _, st = LM.car_clear(_line(2.9), 1, [near], 0.0, max_push=0.3)
    assert st == "drop"


def test_car_clear_trims_in_front_of_the_car_instead_of_dropping():
    x, st = LM.car_clear(_line(2.9), 1, [CAR_R], 0.0, max_push=0.3)  # CPU smoke lane_split R f1-2
    assert st == "trimmed" and not LM.car_inside(x, 1, [CAR_R])
    f = np.isfinite(x)
    assert Z[f].min() == 6.0 and Z[f].max() < CAR_R["zb"] - 0.5 and x[i10] == 2.9  # z 10 kept, untouched
    old = LM.EDGE_CAR_TRIM
    try:
        LM.EDGE_CAR_TRIM = False
        assert LM.car_clear(_line(2.9), 1, [CAR_R], 0.0, max_push=0.3)[1] == "drop"
    finally:
        LM.EDGE_CAR_TRIM = old


def test_push_local_passes_between_rows():
    near = dict(CAR_R); far = {"zb": 25.0, "xl": 5.2, "xr": 7.2, "xa": 5.5, "xb": 6.9}
    x = LM._push_local(_line(3.4), 1, [near, far], None, LM.KERB_CAR_LEN_M)
    assert x is not None and not LM.car_inside(x, 1, [near, far])
    assert abs(x[np.argmin(np.abs(Z - 35.0))] - 3.4) < 1e-6  # far from both cars the line is untouched


def test_edge_car_clear_touches_only_synthetic_edges():
    anch = {"role": "main_R", "side": 1, "xg": _line(3.4), "anchored": True, "held": True}
    live = {"role": "main_R", "side": 1, "xg": _line(3.4)}
    left = {"role": "main_L", "side": -1, "xg": _line(-3.0)}
    out = LM.edge_car_clear([[anch, left], [live, left]], [[CAR_R], [CAR_R]])
    r0 = next(e for e in out[0] if e["role"] == "main_R")
    assert r0.get("car_push") and not LM.car_inside(r0["xg"], 1, [CAR_R])
    assert out[1][0] is live                                   # a live measurement stays as measured
    stuck = dict(anch, xg=_line(2.9))
    old = LM.KERB_MAX_PUSH_M
    try:
        LM.KERB_MAX_PUSH_M = 0.3
        out = LM.edge_car_clear([[stuck, left]], [[CAR_R]])
        r = next(e for e in out[0] if e["role"] == "main_R")  # cannot be pushed -> cut short in front of the car
        assert r["car_push"] and r["zmax"] < CAR_R["zb"] - 0.5 and not LM.car_inside(r["xg"], 1, [CAR_R])
        out = LM.edge_car_clear([[stuck, left]], [[dict(CAR_R, zb=8.0)]])
        assert [e["role"] for e in out[0]] == ["main_L"]      # nothing usable left before the car -> removed
    finally:
        LM.KERB_MAX_PUSH_M = old


def test_synthetic_flags():
    assert LM._synthetic({"anchored": True}) and LM._synthetic({"backfill": True})
    assert LM._synthetic({"held": True, "hold_age": 2})
    assert LM._synthetic({"held": True}) == LM.EDGE_CAR_COAST  # tracker coasts: off by default (A2D2 wjit)
    assert not LM._synthetic({"held": False})


# --- LaneAnchor continuity limit -------------------------------------------------------------------------------------
def test_anchor_rate_limit_is_a_parallel_shift():
    a = LM.LaneAnchor(fps=10.0)
    a.note("main_R", {"xg": _line(7.0)})
    x = a._limit("main_R", _line(9.0, slope=0.05))
    assert abs(x[i10] - (7.0 + LM.EDGE_ANCHOR_RATE / 10.0)) < 1e-9
    f = np.isfinite(x)
    assert np.allclose(np.diff(x[f]), np.diff(_line(9.0, slope=0.05)[f]))
    a.note("main_R", None)
    assert np.array_equal(a._limit("main_R", _line(9.0)), _line(9.0), equal_nan=True)  # no previous output: free


# --- tracker-coast fragment veto ------------------------------------------------------------------------------------
def test_fragment_veto_applies_to_tracker_coasts():
    lanes = [{"id": 1, "side": 1, "xg": _line(1.8)}]
    hook = {"role": "main_R", "side": 1, "xg": np.where((Z >= 12.5) & (Z <= 17.0), 3.2 + 1.1 * (Z - 12.5), np.nan)}
    assert LM.is_diag_fragment(dict(hook, held=True), lanes)                 # BEV tracker coast: vetoed
    assert not LM.is_diag_fragment(dict(hook, held=True, hold_age=1), lanes)  # EdgeHold coast: kept
    old = LM.EDGE_FRAG_COAST
    try:
        LM.EDGE_FRAG_COAST = False
        assert not LM.is_diag_fragment(dict(hook, held=True), lanes)        # iter12 behaviour
    finally:
        LM.EDGE_FRAG_COAST = old


# --- frame-0 anchored backfill -----------------------------------------------------------------------------------------
def test_edge_anchor_backfill_fills_clip_start_from_the_lane_offset():
    lanes = [[{"id": 1, "side": 1, "xg": _line(1.8)}] for _ in range(6)]
    live = {"id": 5, "side": 1, "role": "main_R", "xg": _line(3.3), "zmin": 6.0, "zmax": 40.0}
    frames = [[], [], [live], [live], [live], [live]]
    s = np.arange(6) * 1.0
    out = LM.edge_anchor_backfill(frames, s, 10.0, lanes)
    for j in (0, 1):
        r = next(e for e in out[j] if e["role"] == "main_R")
        assert r["backfill"] and r["anchored"] and abs(r["xg"][i10] - 3.3) < 1e-6
    assert out[2][0] is live


def test_edge_anchor_backfill_off_switch():
    old = LM.EDGE_ANCHOR_BACKFILL
    try:
        LM.EDGE_ANCHOR_BACKFILL = False
        frames = [[], [{"role": "main_R", "side": 1, "xg": _line(3.3)}]]
        assert LM.edge_anchor_backfill(frames, np.zeros(2), 10.0, [[], []]) is frames
    finally:
        LM.EDGE_ANCHOR_BACKFILL = old


# --- front-view rendering: synthetic edges hidden behind parked cars ------------------------------------------------
def test_front_view_hides_synthetic_edge_behind_car():
    import cv2  # noqa: F401
    import bev_smooth as BS
    import foundation_bev as FB
    gm = BS.GroundModel(800.0, 800.0, 640.0, 360.0, 1.5, 0.0)
    img = np.zeros((720, 1280, 3), np.uint8)
    rec = {"xg": _line(4.0, z0=8.0, z1=30.0), "role": "main_R", "side": 1}
    uv = gm.project(np.array([[4.0, 12.0], [4.0, 14.0]]))
    box = [uv[:, 0].min() - 60, uv[:, 1].min() - 80, uv[:, 0].max() + 60, gm.project(np.array([[4.0, 11.0]]))[0, 1]]
    x1, y1, x2, y2 = [int(v) for v in box]
    live = FB.draw_front2(img, gm, 0, [], [], [rec], car_boxes=[box])
    syn = FB.draw_front2(img, gm, 0, [], [], [dict(rec, anchored=True)], car_boxes=[box])
    inside = lambda im: int(im[y1 + 2:y2 - 2, x1 + 2:x2 - 2].any(-1).sum())
    assert inside(live) > 300 and inside(syn) <= 6  # only line-end anti-aliasing at the box bottom
    assert syn.any()  # the rest of the line is still drawn
