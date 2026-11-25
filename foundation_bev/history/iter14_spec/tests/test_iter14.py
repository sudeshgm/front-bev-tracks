"""iter14 unit tests: the sodium-night gate (night_cast) of TB_ADAPT / VPLANE, the adaptive thin-bright threshold
and the road vertical-plane gate behind it, the clip-start backfill car trim knob (FBEV_EDGE_CAR_BF), the spatial
lane-edge divergence candidate (default off) and its time-based confirmation, and the ego14 road-masked phase tier."""
import os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lines_mt as LM
import bev_smooth as BS
import ego_motion as EM
Z = LM.ZS


def _img(road_bgr, top_bgr, h=240, w=320):
    im = np.zeros((h, w, 3), np.uint8)
    im[: int(0.4 * h)] = top_bgr
    im[int(0.4 * h):] = road_bgr
    road = np.zeros((h, w), np.uint8); road[int(0.5 * h):] = 1
    return im, road, int(0.45 * h)


def _hsv2bgr(hh, s, v):
    return cv2.cvtColor(np.uint8([[[hh, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]


# --- night gate ---------------------------------------------------------------------------------------------------
def test_night_cast_sodium_night_on():
    im, road, hv = _img(_hsv2bgr(22, 130, 140), _hsv2bgr(110, 40, 10))
    assert BS.night_cast(im, road, hv)


def test_night_cast_warm_daylight_off():
    # warm daylight asphalt (hue 12, S 90) with a bright sky: the iter14 draft gate fired here, night_cast must not
    im, road, hv = _img(_hsv2bgr(12, 120, 120), _hsv2bgr(105, 30, 200))
    assert not BS.night_cast(im, road, hv)


def test_night_cast_grey_night_off():
    im, road, hv = _img(_hsv2bgr(100, 20, 60), _hsv2bgr(110, 30, 8))  # night, white LED light: no sodium cast
    assert not BS.night_cast(im, road, hv)


def test_night_cast_small_road_off():
    im, road, hv = _img(_hsv2bgr(22, 130, 140), _hsv2bgr(110, 40, 10))
    road[:] = 0; road[-1, :100] = 1
    assert not BS.night_cast(im, road, hv)


def test_thin_bright_default_auto_daylight_unchanged(monkeypatch):
    im, road, hv = _img(_hsv2bgr(12, 120, 120), _hsv2bgr(105, 30, 200))
    im[200:, 150:154] = 255  # a white stripe
    monkeypatch.setenv("FBEV_TB_ADAPT", "0"); m0 = BS.thin_bright_mask(im, road, 0, im.shape[0], hv)
    monkeypatch.delenv("FBEV_TB_ADAPT"); m1 = BS.thin_bright_mask(im, road, 0, im.shape[0], hv)
    assert np.array_equal(m0, m1) and m0.any()


def test_gate_log(tmp_path, monkeypatch):
    p = tmp_path / "g.jsonl"
    monkeypatch.setattr(BS, "_GATE_LOG", str(p))
    im, road, hv = _img(_hsv2bgr(22, 130, 140), _hsv2bgr(110, 40, 10))
    BS.night_cast(im, road, hv, "tb")
    import json
    r = json.loads(p.read_text().splitlines()[0])
    assert r["who"] == "tb" and r["on"] == 1 and r["old"] == 1


# --- VPLANE gate ----------------------------------------------------------------------------------------------------
class _GM:
    fx = fy = 300.0; cx = 160.0; cy = 108.0; h = 1.5


def test_vplane_auto_daylight_noop(monkeypatch):
    import importlib
    fb = importlib.import_module("foundation_bev")
    im, road, hv = _img(_hsv2bgr(12, 120, 120), _hsv2bgr(105, 30, 200))
    depth = np.full(road.shape, 1.0, np.float32)  # everything "vertical"
    monkeypatch.delenv("FBEV_ROAD_VPLANE", raising=False)
    assert np.array_equal(fb.road_vplane_gate(road, depth, im, _GM(), hv), road)


def test_vplane_on_removes_near_vertical(monkeypatch):
    import importlib
    fb = importlib.import_module("foundation_bev")
    im, road, hv = _img(_hsv2bgr(22, 130, 140), _hsv2bgr(110, 40, 10))
    depth = np.full(road.shape, 1.0, np.float32)
    monkeypatch.setenv("FBEV_ROAD_VPLANE", "1")
    out = fb.road_vplane_gate(road, depth, im, _GM(), hv)
    assert out.sum() < road.sum() and out[:, 155:165].sum() == road[:, 155:165].sum()  # the centre column stays


# --- backfill car trim knob -------------------------------------------------------------------------------------------
CAR_R = {"zb": 14.0, "xl": 2.6, "xr": 4.6, "xa": 2.9, "xb": 4.3}


def _line(x0, slope=0.0, z0=6.0, z1=40.0):
    return np.where((Z >= z0) & (Z <= z1), x0 + slope * (Z - 10.0), np.nan)


def test_backfill_trim_knob(monkeypatch):
    e = {"role": "main_R", "side": 1, "xg": _line(3.2), "backfill": True, "held": False}
    o = {"role": "main_L", "side": -1, "xg": _line(-3.5)}
    monkeypatch.setattr(LM, "EDGE_CAR_BF", "trim")
    out = LM.edge_car_clear([[e, o]], [[CAR_R]])[0]
    r = next(x for x in out if x["role"] == "main_R")
    x = np.asarray(r["xg"]); f = np.isfinite(x)
    assert np.nanmax(Z[f]) < CAR_R["zb"] and np.allclose(x[f], 3.2)  # cut before the car, not moved


def test_backfill_push_default():
    assert LM.EDGE_CAR_BF == "push"
    e = {"role": "main_R", "side": 1, "xg": _line(3.2), "backfill": True, "held": False}
    o = {"role": "main_L", "side": -1, "xg": _line(-3.5)}
    out = LM.edge_car_clear([[e, o]], [[CAR_R]])[0]
    r = next(x for x in out if x["role"] == "main_R")
    assert r.get("car_push") and not LM.car_inside(r["xg"], 1, [CAR_R])


# --- spatial divergence (default off) -------------------------------------------------------------------------------
def _lane(x0, sd, slope=0.0):
    return {"id": 1, "side": sd, "xg": _line(x0, slope, 5.0, 40.0)}


def test_spatial_default_off():
    assert not LM.GORE_SPATIAL


def test_spatial_wedge_detected():
    lanes = [_lane(1.8, 1)]
    e = {"role": "main_R", "side": 1, "xg": _line(3.5, 0.08, 5.0, 40.0)}  # opens 2.6 m over 8-40 m
    g = LM.spatial_divergence(lanes, e, 1)
    assert g is not None and g["src"] == "spatial_split"


def test_spatial_parallel_and_kinked_rejected():
    lanes = [_lane(1.8, 1)]
    assert LM.spatial_divergence(lanes, {"xg": _line(5.0, 0.0, 5.0, 40.0)}, 1) is None
    x = _line(3.5, 0.0, 5.0, 40.0); x[Z > 25] += 2.5  # a step (driveway / parking bay), not a wedge
    assert LM.spatial_divergence(lanes, {"xg": x}, 1) is None


def test_spatial_confirmation_is_time_based():
    assert LM.DivergenceGore(fps=4.0).n_spatial == 2
    assert LM.DivergenceGore(fps=10.0).n_spatial == 5
    assert LM.DivergenceGore(fps=30.0).n_spatial == 15


# --- ego14 -----------------------------------------------------------------------------------------------------------
def test_ego_road_default_on():
    import importlib
    fb = importlib.import_module("foundation_bev")
    assert fb.EGO_ROAD


def test_bev_phase_keep_mask_empty_skips():
    gm = BS.GroundModel(300.0, 300.0, 160.0, 108.0, 1.5, 0.0)
    b = EM.BevPhaseEgo(gm, 10.0, 10.0, z_band=(7.0, 40.0), min_resp=0.10)
    im = (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)
    b.step(im, None, keep=np.zeros((240, 320), bool)); b.step(im, None, keep=np.zeros((240, 320), bool))
    assert b.keep_frac == 0.0
