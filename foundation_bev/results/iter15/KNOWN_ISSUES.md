# iter15 known issues (confirmed on the T4 GPU run, 25 Nov 2026)

1. **Lane-pitch speed correction is not uniform (KITTI OXTS):** late -9.2 -> +0.6 %, cyclist +10.7 -> +6.1 %,
   fc1_split +13.2 -> +1.0 % (VO; label trajectory +15.5 -> +3.1 %), but exit -3.4 -> +4.6 % and final -1.3 -> +3.9 %.
   KITTI mean |err| 7.6 -> 6.0 %. Off switch `FBEV_EGO_PITCH=0`. Clips with < 5 lane pairs are untouched (lane_split,
   fc1_merge, a2d2a/b): lane_split stays -18 % (11 deg/s turn; pure-translation phase under-reads).
2. **Lines see the corrected speed in pitch-0 BEV metres.** Only the speed is corrected; the ground model of the lines is
   unchanged, so line motion compensation mixes true travel with BEV range that is off by (1 + theta z / h).
   Measured effect: +-1-6 mm wjit on public (exit edge .042 -> .040, late edge .035 -> .041), +-1-3 cm on fc1_split.
3. **Yaw on a2d2 / munich has no GT.** a2d2a/b are dashboard renders and munich is a dashcam. Only the sign (unit
   test through ego_to_world) and the KITTI / fc1 references back the BEV-ECC yaw; on those clips the drawn ego path
   and the exported ego psi change, lines and tracks do not.
4. **fc1_split yaw is still noisy** (VO corr 0.51, MAE 1.1 deg/s, night); heading drift -2.0 deg over 345 m.
5. **The KITTI cams use the 2011_09_26 P2 for the 4 highway clips, which are 2011_10_03 drive 0042** (cy 185.2 vs
   172.9 px). Swapping the intrinsics makes the speed worse (+19..+25 % vs OXTS), so the cams are unchanged; the
   lane pitch absorbs the effective horizon.
6. Carried over from iter14: city edge wjit 0.054 (`FBEV_EDGE_CAR_BF=trim` restores 0.037, off), night-gate margin on
   late, split fork 0/6 and merge gores 0/26 on fc1, fc1_merge right edge 6.4 m off at z10, frame-0 gaps
   (lane_split R, munich R edge, a2d2), WildDet3D truck misses, kerb-pushed edges over car footprints (munich, a2d2a).
7. **munich lane pitch 0.945 deg is measured and re-passed but unused** (BEV phase rejected for repeated frames; speed
   source scene). Output-neutral, wasted re-decode, and the theta is implausible for that dashcam. Follow-up: skip the
   re-pass when the BEV phase cannot be selected.
8. **Stage-1 WildDet3D is not bit-reproducible on GPU** for munich / a2d2a / a2d2b (iter13 differed from iter12 = iter15
   by 9-10 boxes). Compare tracks only on identical dets; the a2d2a static low-score bicycle id 32 and the a2d2b person
   id 13/14 flip with it.
