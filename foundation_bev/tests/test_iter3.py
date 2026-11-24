import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import bev_smooth as BS, bev_gate as G, ego_motion as EM

gm = BS.GroundModel(371.8, 923.6, 314.1, 221.3, 1.65, 0.0)
# score + ground contact
assert not G.score_ok("bicycle", 0.4) and G.score_ok("car", 0.4)
z = 20.0; vb = 221.3 + 923.6 * 1.65 / z
ok, why, zg = G.ground_contact((300, vb - 1.7 * 923.6 / z, 300 + 0.6 * 371.8 / z, vb), "person", gm, depth_z=20.5)
assert ok, why
assert G.ground_contact((300, 100, 320, 200), "person", gm)[1] == "bottom_above_horizon"
assert G.ground_contact((300, vb - 1.7 * 923.6 / z, 310, vb), "person", gm, depth_z=35)[1] == "depth_vs_ground"
assert G.ground_contact((300, vb - 6 * 923.6 / z, 310, vb), "person", gm)[1] == "too_tall"
# vertical plane: a flat wall at z = 12 m fills the image -> person box on it is rejected; ground scene is not
H, W = 480, 640
wall = np.full((H, W), 12.0)
assert G.on_vertical_plane((300, 150, 340, 260), wall, 371.8, 923.6, 314.1, 221.3)
vv = np.arange(H)[:, None] * np.ones((1, W))
ground = np.where(vv > 230, 923.6 * 1.65 / np.maximum(vv - 221.3, 1), 80.0)
scene = ground.copy(); scene[300:420, 300:340] = 923.6 * 1.65 / (420 - 221.3)  # upright person, ground behind
assert not G.on_vertical_plane((300, 300, 340, 420), scene, 371.8, 923.6, 314.1, 221.3)
# bicycle needs a rider or size prior
dets = [dict(box=(300, vb - 1.6 * 923.6 / z, 300 + 0.5 * 371.8 / z, vb), cls="bicycle", score=0.7)]
assert len(G.gate_detections(dets, gm)[0]) == 0
dets.append(dict(box=(298, vb - 1.8 * 923.6 / z, 302 + 0.5 * 371.8 / z, vb), cls="person", score=0.7))
assert len(G.gate_detections(dets, gm)[0]) == 2
# track filter: static parked car vs moving car, short stub
fps, v = 10.0, 10.0
ego = {"s": np.arange(60) * v / fps}
fr = list(range(40))
tr = {"parked": dict(cls="car", frames=fr, x=[5.0] * 40, z=[45 - v * f / fps for f in fr], scores=[0.8] * 40, hits=40),
      "lead": dict(cls="car", frames=fr, x=[0.0] * 40, z=[20.0] * 40, scores=[0.8] * 40, hits=40),
      "stub": dict(cls="car", frames=fr[:3], x=[5.0] * 3, z=[30, 29, 28], scores=[0.8] * 3, hits=3),
      "bike": dict(cls="bicycle", frames=fr, x=[8.0] * 40, z=[30.0] * 40, scores=[0.9] * 40, hits=40)}
d = G.track_filter(tr, fps, ego)
assert d == {"parked": "draw_static", "lead": "draw", "stub": "drop_short", "bike": "drop_bicycle_no_rider"}, d
prof = G.scene_ego_profile(tr, 60, fps, prior=8.0, min_n=1)
assert abs(np.median(prof["speed"]) - 10.0) < 0.5, prof["speed"][:5]
# rigid 2D RANSAC recovers yaw + forward motion
rng = np.random.default_rng(1)
A = np.stack([rng.uniform(-6, 6, 200), rng.uniform(5, 25, 200)], 1)
th, ds = 0.02, 1.5
R = np.array([[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]])
B = A @ R.T + np.array([0, -ds]); B[:40] += rng.uniform(-3, 3, (40, 2))
r = EM.rigid2d_ransac(A, B)
assert abs(-r[0] - th) < 2e-3 and abs(-r[1][1] - ds) < 0.05, r[:2]
# ego path: constant yaw rate -> curved path ahead
e = {"x": np.zeros(5), "z": np.arange(5.0), "psi": np.zeros(5), "speed_mps": np.full(5, 10.0), "yaw_rate": np.full(5, 0.2)}
beh, ah = EM.ego_path_in_current_frame(e, 4)
assert ah[-1, 0] > 1.0 and len(beh) == 5
print("ALL ITER3 TESTS PASSED")
