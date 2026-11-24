# iter13 known issues (GPU run 25 Nov 2026: items 1-6 confirmed, see metrics_vs_iter12.md)

1. **City edge wjit 0.037 → 0.054, jit2 L 0.021 → 0.023** (replay vs the iter12 GPU run); a2d2a edge jit2 L 0.035 → 0.037 (GPU).
   The f0-1 backfilled L edge is pushed out of the parked-car row while the first live edges still run along it.
   `FBEV_EDGE_CAR_CLEAR=0` restores iter12.
2. **Kerb-pushed measured edges can still overlap car footprints** (munich 63, a2d2a 44 frames in the audit). iter13 only
   clears synthetic edges (anchored / backfilled / EdgeHold); measured edges are left as measured.
3. **munich f200 left edge** sits ~0.5 m beyond the kerb on the tram verge (the orange L edge next to the green L lane).
   Right edge at f200 is now drawn behind the parked cars (front-view occlusion); it was ~1.3 m beyond their outer flank,
   not inside them. The genuinely inside-car anchored frames were R190-196, 250-251 (fixed).
4. **munich R z10 f615 lost**: a tracker-coasted diagonal fragment, now vetoed (`FBEV_EDGE_FRAG_COAST=1`).
5. **Frame 0 still incomplete** on lane_split (R lane), munich (R edge), a2d2a (L lane), a2d2b (lanes): no lane
   evidence at f0; not fixable cheaply without inventing lines.
6. **lane_split lane wjit 0.131** (YOLO 0.092): structural, unchanged.
7. **Private clips** (evaluated locally only): merge / split gores, night road-edge overshoot and 4 fps ego speed
   are open; aggregate follow-up is iter14 work.
