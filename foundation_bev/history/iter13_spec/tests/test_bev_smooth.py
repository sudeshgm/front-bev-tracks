"""CPU unit tests for bev_smooth (synthetic data). Run: python3 tests/test_bev_smooth.py"""
import math, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import bev_smooth as B


def test_ground_roundtrip():
    for pitch in (0.0, math.radians(2), math.radians(-1.5)):
        g = B.GroundModel(700, 700, 320, 200, 1.6, pitch)
        xz = np.array([[-3.5, 8.0], [0.0, 20.0], [2.0, 35.0]])
        uv = g.project(xz)
        back = g.lift(uv[:, 0], uv[:, 1])
        assert np.allclose(back, xz, atol=1e-6), (pitch, back)
    g = B.GroundModel(371.8, 923.6, 314.1, 221.3, 1.65, 0.0)
    # flat model equals z = fy*H/(v-cy)
    z = g.lift([314.1], [221.3 + 923.6 * 1.65 / 10.0])[0, 1]
    assert abs(z - 10.0) < 1e-6
    print("ground roundtrip ok")


def test_plane_pitch():
    fx = fy = 500.0; cx, cy = 320.0, 240.0; H = 1.5; p = math.radians(2.0)
    g = B.GroundModel(fx, fy, cx, cy, H, p, zmax=60)
    h, w = 480, 640
    vs, us = np.mgrid[300:470, 100:540]
    xz = g.lift(us.ravel(), vs.ravel())
    # camera-z depth of each ground point
    X, Z = xz[:, 0], xz[:, 1]
    zc = H * math.sin(p) + Z * math.cos(p)
    depth = np.zeros((h, w)); depth[vs.ravel(), us.ravel()] = zc
    road = (depth > 0).astype(np.uint8)
    pitch, height = B.dav2_plane_pitch(depth, road, fx, fy, cx, cy, (300, 470))
    assert abs(math.degrees(pitch) - 2.0) < 0.1 and abs(height - H) < 0.02, (math.degrees(pitch), height)
    print("dav2 plane pitch ok", round(math.degrees(pitch), 3), round(height, 3))


def synth_tracks(seed=0, n=60, fps=10.0, speed=10.0):
    rng = np.random.default_rng(seed)
    dt = 1 / fps
    frames = []
    # car A: overtaking, world speed 14 m/s, lateral 3.5; car B parked at x=-4, s=60; person crossing
    for k in range(n):
        t = k * dt; trav = speed * t
        dets = []
        sA = 15 + 14 * t; zA = sA - trav
        if not (20 <= k <= 23):  # gap of 4 frames (occlusion)
            dets.append(("car", 3.5, zA, 0.85))
        sB = 60.0; zB = sB - trav
        if zB > 2:
            dets.append(("car", -4.0, zB, 0.6 if k % 3 else 0.4))  # low-score frames exercise stage 2
        xP = -2 + 1.2 * t
        sP = 30.0; zP = sP - trav
        if zP > 2:
            dets.append(("person", xP, zP, 0.7))
        out = []
        for cls, x, z, sc in dets:
            lat, lon = B.meas_sigma(z)
            mx, mz = x + rng.normal(0, lat), z + rng.normal(0, lon)
            u = 320 + 500 * x / max(z, 1); v = 240 + 500 * 1.5 / max(z, 1)
            bw = 500 * (1.8 if cls == "car" else 0.6) / max(z, 1)
            out.append({"cls": cls, "score": sc, "box": [u - bw / 2, v - bw, u + bw / 2, v], "bev": (mx, mz), "src": "ground"})
        frames.append(out)
    return frames


def accel_rms(pos, dt):
    pos = np.asarray(pos)
    if len(pos) < 3:
        return float("nan")
    a = np.diff(pos, 2, axis=0) / dt / dt
    return float(np.sqrt(np.mean(np.sum(a * a, 1))))


def test_tracker():
    fps, speed = 10.0, 10.0
    T = B.ObjTracker(fps, speed)
    frames = synth_tracks(fps=fps, speed=speed)
    for k, dets in enumerate(frames):
        T.step(k, dets)
    tracks = T.finalize()
    cls = sorted(t["cls"] for t in tracks)
    assert cls.count("car") == 2 and cls.count("person") == 1, cls
    for t in tracks:
        raw = [r["meas"] for r in t["rows"] if r["meas"] is not None]
        sm = [r["smooth"] for r in t["rows"]]
        ar, asm = accel_rms(raw, 1 / fps), accel_rms(sm, 1 / fps)
        print(f"track {t['id']} {t['cls']} len={len(t['rows'])} stationary={t['stationary']} accelRMS raw={ar:.1f} smooth={asm:.2f}")
        assert asm < 0.25 * ar or t["stationary"]
    print("tracker ok, relinks:", T.n_relink)


def test_relink():
    fps, speed = 10.0, 10.0
    T = B.ObjTracker(fps, speed, coast=2)  # short coast forces a fragment -> re-link must join it
    frames = synth_tracks(fps=fps, speed=speed)
    for k, dets in enumerate(frames):
        T.step(k, dets)
    tracks = T.finalize()
    cars_moving = [t for t in tracks if t["cls"] == "car" and not t["stationary"]]
    assert len(cars_moving) == 1, [(t["id"], len(t["rows"])) for t in tracks]
    print("relink ok, relinks:", T.n_relink)


def test_lines():
    rng = np.random.default_rng(1)
    ds = 1.0
    L = B.LineTracker("lane")
    xs10 = []
    for k in range(40):
        pts = []
        for c0 in (-1.75, 1.75):
            z = rng.uniform(5, 35, 60)
            x = c0 + 0.002 * z * z + rng.normal(0, 0.08, z.size)
            pts.append(np.stack([x, z], 1))
        if k == 15:
            pts.append(np.stack([np.full(30, 0.6), np.linspace(5, 30, 30)], 1))  # outlier stroke
        allp = np.concatenate(pts)
        if k in (20, 21, 22, 23, 24):
            allp = allp[:0]  # 5 missing frames -> hold
        fits = [f for f in (B.ransac_poly2(c) for c in B.cluster_lateral(allp)) if f]
        live = L.step(k, fits, ds)
        right = [r for r in live if r["c"][0] > 0]
        if right:
            xs10.append(B.x_at(right[0]["c"], 10.0))
    assert len(xs10) >= 36, len(xs10)  # held through the gap
    j = np.median(np.abs(np.diff(xs10)))
    assert j < 0.05, j
    sm = B.centred_smooth(L.hist[1] if 1 in L.hist else next(iter(L.hist.values())), ds)
    print("lines ok: right lane x(10) median jitter %.3f m, live=%d, gated=%d, smoothed frames=%d" % (j, len(live), L.n_gated, len(sm)))


def test_thin():
    m = np.zeros((60, 60), np.uint8)
    m[10:50, 25:31] = 1
    s = B.thin(m)
    assert s.sum() > 20 and s[:, :].sum(1).max() <= 2
    print("thin ok", int(s.sum()))


if __name__ == "__main__":
    test_ground_roundtrip(); test_plane_pitch(); test_tracker(); test_relink(); test_lines(); test_thin()
    print("ALL TESTS PASSED")
