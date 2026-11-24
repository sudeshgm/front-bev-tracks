import json, sys
J = json.load(open(sys.argv[1]))
idx = {r: {d['clip']: d for d in J[r]} for r in J}
print("| clip | edge L/R pres: iter5 (no gate) | iter6 rule (replay) | iter6 GPU | **iter7 rule** | lane L/R pres iter5 → iter7 | edge L/R jit2 iter5 → iter7 |")
print("|---|---|---|---|---|---|---|")
for c in idx['iter5_raw']:
    g = lambda r, k: idx[r].get(c, {}).get(k)
    e = lambda r, a='edges', s='pres': f"{g(r, a + '_L10_' + s)}/{g(r, a + '_R10_' + s)}"
    print(f"| {c} | {e('iter5_raw')} | {e('iter6_rule')} | {e('iter6_gpu')} | **{e('iter7_rule')}** | {e('iter5_raw','lanes')} → {e('iter7_rule','lanes')} | {e('iter5_raw','edges','jit2')} → {e('iter7_rule','edges','jit2')} |")
