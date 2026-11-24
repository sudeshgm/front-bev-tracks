# iter12 known issues

Closed by iter12 (GPU run, 24 Nov 2026): gores still fire only on exit approach f21–26 and Munich f760–765; idsw is 0;
tracks unchanged. Edge width jitter on city 0.076 → 0.037 and late 0.041 → 0.034 (iter11 regression closed; the
live-only value is unchanged at 0.054 / 0.035). Munich right edge at z = 10 m 777 → 862/900 (+85, none lost),
f165–166 present. a2d2a frame-0 right edge present.

Open (iter13 targets):
- **Munich f189–251: the anchored right edge is drawn over the parked cars.** At f200 (see `key_stills/munich_f0200.jpg`)
  the BEV line is at x ≈ 8.6–9.0 m, about 1.3 m beyond the silver car's outer flank. But its whole visible part
  (z ≥ 13 m) projects inside that car's image box, because the ground there is hidden behind the car, so in the
  camera view it runs along the car bodies. LaneAnchor output also never gets the kerb push. Worse, on f190–196 and
  f250–251 the anchored line jumps inward to 6.8–7.1 m, inside the car footprints, whenever the same-side lane drops
  out and the reference flips to the opposite edge (x at z 10 m swings between 6.8 and 9.8 m over f190–250).
  Across all clips, 14 Munich and 1 a2d2b anchored frames, plus 5 backfilled frames, lie inside a car footprint.
- **Munich R f562–563 and f570–571 absent** (expected present). On exactly these frames the BEV tracker coasts the raw
  edge, and the iter11 fragment veto skipped held records, so the coasted forecourt hook (x 3.2 → 8.6 m over z 10–15 m)
  survived as a z 12.5–17 m fragment.
- **Munich f200 left edge:** there is no painted line on the left. Lane 180 (−2.26 m at z 10) is the concrete kerb
  between the road and the grass/tram bed. main_L (−2.80 m) lies about 0.5 m further out, on the grass-verge/tram-bed
  boundary, so the road edge is about 0.5 m too wide there. It is not on the rails.
- **Frame 0 still incomplete:** lane_split (right lane: only an unconfirmed parking-bay line), Munich (right edge: the only
  copy is kinked), a2d2a (left lane), a2d2b (lanes; the left edge copy is diagonal).
- **lane_split lane width jitter 0.131** (YOLOPv2 0.092): structural (nearest-pair identity switches through the split).
