# iter10 known issues

- **False widen gore on Munich f881–885 (fixed in iter11).** The GPU run fires a `widen` gore on Munich f881–885 that iter9 did not have. These frames have full-range main_L/main_R edges, and main_R is `kerb_occ: true` on all five frames in iter10 (only f881–882 in iter9). The expected gores still fire: exit approach f21–26 (widen) and Munich f760–765 (fork). Root cause and fix: see `SPEC.md` of iter11.
- **Munich right edge at z = 10 m: +19 frames, not the +26 expected (757 → 776 / 900).** Gained f162–164, f167–188, f256, f258, f266–267 and f618; lost f520–526 and f557–560; f165–166 are still missing.
- **Road edges are not backfilled on the first frames** (for example exit f0); only lanes are (iter11).
