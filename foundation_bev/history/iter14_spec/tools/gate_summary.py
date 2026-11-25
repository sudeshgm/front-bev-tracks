"""iter14: per-clip night-gate summary of a FBEV_GATE_LOG file (gate.jsonl). Prints a markdown table:
clip | gate calls | on (must be 0 on public clips) | draft-gate passes (S>70 & hue) | closest call (max S, min top V)."""
import json, os, sys
from collections import defaultdict
rows = defaultdict(list)
for path in sys.argv[1:]:  # one combined gate.jsonl, or one <clip>.jsonl per clip (tag falls back to the file name)
    for line in open(path):
        if line.strip():
            r = json.loads(line); rows[r.get("tag") or os.path.basename(path).rsplit(".", 1)[0]].append(r)
print("| clip | gate calls | on | draft-gate passes | road S max | top V min |")
print("|---|---|---|---|---|---|")
tot = 0
for t in sorted(rows):
    L = rows[t]; on = sum(r["on"] for r in L); tot += on
    print(f"| {t} | {len(L)} | **{on}** | {sum(r['old'] for r in L)} | {max(r['S'] for r in L):.0f} | {min(r['topV'] for r in L):.0f} |")
print(f"\nTOTAL on = {tot} ({'OK: TB_ADAPT / VPLANE never ran' if tot == 0 else 'GATE FIRED: rerun those clips with FBEV_TB_ADAPT=0 FBEV_ROAD_VPLANE=0 and diff'})")
