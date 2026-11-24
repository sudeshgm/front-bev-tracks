import json, sys
J = json.load(open(sys.argv[1])); A, B = (sys.argv[2:4] if len(sys.argv) > 3 else list(J)[:2])
ix = {r: {d['clip']: d for d in J[r]} for r in (A, B)}
print(f"| clip | accel_rms {A} → {B} | edge L/R pres | lane L/R pres | idsw/100f |")
print("|---|---|---|---|---|")
for c in ix[A]:
    a, b = ix[A][c], ix[B].get(c, {})
    p = lambda d, k: f"{d.get(k+'_L10_pres')}/{d.get(k+'_R10_pres')}"
    print(f"| {c} | {a['accel_rms_mps2']} → {b.get('accel_rms_mps2')} | {p(a,'edges')} → {p(b,'edges')} | {p(a,'lanes')} → {p(b,'lanes')} | {a.get('id_switch_proxy_per100f')} → {b.get('id_switch_proxy_per100f')} |")
