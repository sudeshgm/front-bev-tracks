"""iter15: what changed between two runs per clip: tracks csv identical?, polylines identical apart from the per-frame
'ego' pose?, number of frames whose lanes / edges differ, ego speed before -> after. usage: diff_scope.py OLD NEW clip..."""
import sys, json, filecmp
old, new = sys.argv[1:3]
for c in sys.argv[3:]:
    try:
        a = json.load(open(f"{old}/polylines/{c}_polylines.json")); b = json.load(open(f"{new}/polylines/{c}_polylines.json"))
    except FileNotFoundError:
        print(c, "missing"); continue
    tr = filecmp.cmp(f"{old}/tracks/{c}_tracks.csv", f"{new}/tracks/{c}_tracks.csv", shallow=False)
    fa, fb = a.pop("frames"), b.pop("frames")
    ego_d = sum(1 for x, y in zip(fa, fb) if x.get("ego") != y.get("ego"))
    for x in fa + fb: x.pop("ego", None)
    line_d = sum(1 for x, y in zip(fa, fb) if x != y)
    for d in (a, b): d.pop("version", None); d.pop("ego_speed_mps", None)
    hdr = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
    ea = json.load(open(f"{old}/results/{c}.json"))["ego_speed"]["speed_median_mps"]; eb = json.load(open(f"{new}/results/{c}.json"))["ego_speed"]["speed_median_mps"]
    print(f"{c:30s} speed {ea:6.2f} -> {eb:6.2f} | tracks {'identical' if tr else 'DIFF'} | ego pose differs on {ego_d}/{len(fb)} frames | lanes/edges differ on {line_d}/{len(fb)} frames | header keys changed: {hdr}")
