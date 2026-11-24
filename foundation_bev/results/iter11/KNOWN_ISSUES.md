# iter11 known issues

Closed by iter11 (GPU run, 24 Nov 2026): the iter10 false widen gore on Munich f881–885 is gone. Gores fire only on exit approach f21–26 and Munich f760–765. idsw is 0 on every clip, and road edges are present on 100 % of frames on every KITTI clip.

Status after iter12 (GPU run, 24 Nov 2026):
- CLOSED: city / late edge width jitter (0.076 → 0.037, 0.041 → 0.034).
- CLOSED: Munich f165–166 (present). Munich R 777 → 862/900; f561, f564–569 and f576–577 recovered.
- PARTLY: Munich f562–563 / f570–571 still absent (tracker-coasted fragment, see results/iter12/KNOWN_ISSUES.md).
- PARTLY: frame 0: a2d2a right edge fixed; lane_split, Munich and a2d2b remain.
- OPEN: lane_split lane width jitter (structural).

Open at iter11 (iter12 targets):
- **Edge width jitter rose** on city (+0.022) and late (+0.006). The clip-start edge backfill adds frames that continue the start-of-clip width drift.
- **lane_split lane width jitter is 0.131** (YOLOPv2 baseline 0.092).
- **Munich right edge at z = 10 m: 777/900.** Gained f520–526, f550–560 and f615. Lost f561–577 and f618 (iter10 drew a "<" zigzag there). f165–166 are still missing.
- **Munich held-edge frames went from 189 to 221.**
- **Frame 0 is incomplete** on four clips:
  - lane_split: no right lane;
  - Munich: main_R starts at z = 17 m;
  - a2d2a;
  - a2d2b.
