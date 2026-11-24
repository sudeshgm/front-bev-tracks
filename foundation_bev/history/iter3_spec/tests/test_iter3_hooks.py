import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import foundation_bev as FB

n, fps = 20, 10.0
vo = {"speed_mps": np.full(n, 10.5), "speed_median_mps": 10.5, "valid_frac": 0.9}
sc = {"speed": np.full(n, 10.0), "valid_frac": 0.8}
assert FB.select_ego_speed({}, vo, sc, 25.0, n, fps)["source"] == "flow"            # consistent flow wins
assert FB.select_ego_speed({}, dict(vo, speed_median_mps=3.0), sc, 25.0, n, fps)["source"] == "scene"  # inconsistent flow
assert FB.select_ego_speed({}, dict(vo, valid_frac=0.2), dict(sc, valid_frac=0.0), 25.0, n, fps)["source"] == "prior"
r = FB.select_ego_speed({"oxts_speed_mps": 7.0}, vo, sc, 25.0, n, fps)
assert r["source"] == "oxts" and abs(r["s"][-1] - 7.0 * (n - 1) / fps) < 1e-6
# dashed aggregation: two half-dashes 3 m apart become one point set in the current frame
agg = FB.LaneAggregator(3)
a = agg.push(np.array([[1.8, 10.0], [1.8, 11.0]]), 0.0)
b = agg.push(np.array([[1.8, 14.0], [1.8, 15.0]]), 2.0)
assert len(b) == 4 and np.allclose(sorted(b[:, 1]), [8.0, 9.0, 14.0, 15.0]), b
print("ALL ITER3 HOOK TESTS PASSED")
