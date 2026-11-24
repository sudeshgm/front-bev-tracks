"""iter4 multi-line BEV tracking with forks, gore areas and widening road edges.

Replaces iter2/3's "nearest left/right lane + one edge per side, each one global x(z) poly, EMA with an
absolute 0.5 m gate" (which averages or drops a diverging exit line).

Representation: every lane line and every edge is a polyline x(z) sampled on a fixed z grid (0.5 m),
valid only over its observed span, so a line may bend, diverge or end anywhere.

Per frame (BEV masks in, from SAM 2.1 paint/road masks on GPU or bev_ipm on CPU):
  extract_chains      paint skeleton -> long near-longitudinal chains (lines) and short oblique stripes
                      (gore chevrons); chains link across dash gaps along their own heading
  smooth_chain        smoothing spline x(z) (scipy UnivariateSpline k=2) -> sampled polyline
  edge_chains         per BEV row: runs of the drivable mask; the run under the ego gives the main
                      left/right boundary, a second run (exit / on-ramp) gives a split edge pair
  gore_polygons       clusters of >= 2 parallel oblique stripes -> convex hull polygon
Across frames:
  MultiLineTracker    Hungarian on lateral distance at many z over the overlap + heading term, gated on
                      each line's own innovation scale; per-z update (no global poly), birth after
                      n_birth hits, death after n_death misses, no fixed slot count; a newborn line that
                      touches an existing line (< 1 m) at one end becomes its child (fork / merge point)
  GoreTracker         hit-count + EMA on polygon vertices
"""
import json
import os

import numpy as np

# iter13: low-frame-rate time base (fc1_merge is 4 fps). Frame-count constants were tuned at 10 Hz (KITTI, A2D2)
# and kept as-is on Munich (30 Hz). Below 10 fps they are scaled to the same TIME span as at 10 Hz, with a floor,
# so clips at >= 10 fps are bit-identical. FBEV_LOWFPS_TIME=0 restores raw frame counts.
LOWFPS_TIME = os.environ.get("FBEV_LOWFPS_TIME", "1") == "1"


def fr10(n, fps, floor=1):
    """Frame count n tuned at 10 Hz -> same time span at fps when fps < 10 (>= floor); unchanged at >= 10 fps."""
    if not LOWFPS_TIME or fps is None or float(fps) >= 10.0:
        return n
    return max(int(floor), int(round(n * float(fps) / 10.0)))

# iter5b: NumPy 1.x has only np.trapz; NumPy >= 2.? removed np.trapz (2.0 added np.trapezoid)
_trapz = getattr(np, "trapezoid", None) or np.trapz
import cv2
from scipy.interpolate import UnivariateSpline
from scipy.optimize import linear_sum_assignment

ZS = np.arange(5.0, 45.01, 0.5)


# ---------------------------------------------------------------------------------------------- per frame

def _skeleton(mask):
    m = (mask > 0).astype(np.uint8)
    try:
        return cv2.ximgproc.thinning(m * 255) > 0
    except Exception:  # noqa: BLE001
        sk = np.zeros_like(m, bool)
        el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = m.copy()
        while img.any():
            er = cv2.erode(img, el)
            sk |= (img - cv2.dilate(er, el)) > 0
            img = er
        return sk


def components(paint, res, min_len_m=0.6):
    """Connected paint components with PCA geometry in metres (x right, z forward).
    Returns list of dict(pts (N,2) x,z of skeleton, len, ang_from_z (rad), centre)."""
    from bev_ipm import bev_px_to_xz
    n, lab, stats, _ = cv2.connectedComponentsWithStats(paint.astype(np.uint8), connectivity=8)
    sk = _skeleton(paint)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 6:
            continue
        vv, uu = np.nonzero((lab == i) & sk)
        if len(uu) < 4:
            vv, uu = np.nonzero(lab == i)
        x, z = bev_px_to_xz(uu, vv)
        P = np.stack([x, z], 1)
        c = P.mean(0)
        if len(P) < 3:
            continue
        w, V = np.linalg.eigh(np.cov((P - c).T))
        d = V[:, 1]
        proj = (P - c) @ d
        L = float(proj.max() - proj.min())
        if L < min_len_m:
            continue
        ang = float(np.arccos(min(1.0, abs(d[1]))))
        width = float(2 * np.sqrt(max(w[0], 0)))
        out.append({"pts": P, "len": L, "ang": ang, "centre": c, "dir": d if d[1] >= 0 else -d, "width": width})
    return out


def extract_chains(comps, max_gap_m=9.0, lat_tol=0.35, max_line_ang=np.radians(35), min_chain_len=4.0):
    """Link near-longitudinal components into chains along their heading (dash gaps). Oblique short
    components are returned separately as gore stripe candidates."""
    stripes = [c for c in comps if c["ang"] > np.radians(20) and 0.6 <= c["len"] < 4.0]
    sid = {id(c) for c in stripes}
    lines = [c for c in comps if id(c) not in sid and c["ang"] <= max_line_ang and c["len"] >= 1.0]
    lines.sort(key=lambda c: c["pts"][:, 1].min())
    chains = []
    used = [False] * len(lines)
    for i, c in enumerate(lines):
        if used[i]:
            continue
        used[i] = True
        chain = [c]
        while True:
            last = chain[-1]
            top = last["pts"][np.argmax(last["pts"][:, 1])]
            # local heading from the chain so far (slope dx/dz)
            P = np.concatenate([q["pts"] for q in chain])
            slope = np.polyfit(P[:, 1], P[:, 0], 1)[0] if np.ptp(P[:, 1]) > 1.0 else last["dir"][0] / max(last["dir"][1], 1e-3)
            best, bd = None, None
            for j, d in enumerate(lines):
                if used[j]:
                    continue
                bot = d["pts"][np.argmin(d["pts"][:, 1])]
                gap = bot[1] - top[1]
                if gap < -0.5 or gap > max_gap_m:
                    continue
                pred = top[0] + slope * gap
                e = abs(bot[0] - pred)
                dslope = abs(d["dir"][0] / max(d["dir"][1], 1e-3) - slope)
                if e < lat_tol + 0.04 * gap and dslope < 0.25:
                    sc = e + 0.05 * gap
                    if bd is None or sc < bd:
                        best, bd = j, sc
            if best is None:
                break
            used[best] = True
            chain.append(lines[best])
        P = np.concatenate([q["pts"] for q in chain])
        if np.ptp(P[:, 1]) >= min_chain_len or (len(chain) == 1 and chain[0]["len"] >= 2.5):
            chains.append(P)
    return chains, stripes


def smooth_chain(P, zs=ZS, smooth=0.04, max_extrap=0.0):
    """Smoothing spline x(z) through chain points -> x sampled on zs (NaN outside the span)."""
    P = P[np.argsort(P[:, 1])]
    # collapse duplicate z (spline needs strictly increasing z)
    zb = np.round(P[:, 1] / 0.25) * 0.25
    uz, inv = np.unique(zb, return_inverse=True)
    ux = np.array([np.median(P[inv == k, 0]) for k in range(len(uz))])
    x = np.full(len(zs), np.nan)
    if len(uz) < 3:
        return x
    k = 2 if len(uz) >= 5 else 1
    try:
        sp = UnivariateSpline(uz, ux, k=k, s=len(uz) * smooth ** 2)
    except Exception:  # noqa: BLE001
        return x
    m = (zs >= uz[0] - max_extrap) & (zs <= uz[-1] + max_extrap)
    x[m] = sp(zs[m])
    return x


def clean_road(road, res, close_len_m=3.0):
    """Drivable mask -> one solid region: longitudinal closing (bridges dash/shadow gaps), hole filling
    (gore hatching, shadows, dashes inside the road stay drivable) and the largest component."""
    m = (road > 0).astype(np.uint8)
    k = max(3, int(round(close_len_m / res)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m > 0
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8)
    inv = (1 - m).astype(np.uint8)
    ff = inv.copy(); mask = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)
    for seed in ((0, 0), (m.shape[1] - 1, 0), (0, m.shape[0] - 1), (m.shape[1] - 1, m.shape[0] - 1)):
        if ff[seed[1], seed[0]]:
            cv2.floodFill(ff, mask, seed, 2)
    holes = (ff == 1)
    small = np.zeros_like(holes)
    nh, lh, sh, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8))
    for i in range(1, nh):  # fill holes < 25 m^2 (shadows, hatching); keep big islands (gore grass, split)
        if sh[i, cv2.CC_STAT_AREA] * res * res < 25.0:
            small |= lh == i
    return (m > 0) | small


def edge_chains(road, res, zs=ZS, min_run_m=1.2, med=5, valid=None):
    """Drivable-area boundary per BEV row. Returns dict of sampled x arrays:
    main_L, main_R (run under / nearest the ego), and for a second run on either side (fork, exit lane,
    on-ramp) split_in / split_out (its inner and outer boundaries)."""
    from bev_ipm import xz_to_bev_px, bev_px_to_xz
    H, W = road.shape
    road = clean_road(road, res)
    out = {k: np.full(len(zs), np.nan) for k in ("main_L", "main_R", "split_in", "split_out")}
    side = np.zeros(len(zs))
    _, v = xz_to_bev_px(np.zeros(len(zs)), zs)
    u0 = xz_to_bev_px(0.0, 0.0)[0]
    prev_c = u0
    for i, vv in enumerate(np.round(v).astype(int)):
        if not (0 <= vv < H):
            continue
        row = road[vv]
        cols = np.flatnonzero(row)
        if cols.size == 0:
            continue
        splits = np.where(np.diff(cols) > 2)[0] + 1
        runs = [(r[0], r[-1]) for r in np.split(cols, splits) if (r[-1] - r[0]) * res >= min_run_m]
        if not runs:
            continue
        # main run: contains (or is nearest to) the previous row's main centre
        mi = int(np.argmin([0 if a <= prev_c <= b else min(abs(a - prev_c), abs(b - prev_c)) for a, b in runs]))
        a, b = runs[mi]
        prev_c = 0.5 * (a + b)
        # a boundary that touches the image/IPM border is the camera's field of view, not a road edge
        touchL = valid is not None and (a <= 3 or not valid[vv, max(0, a - 4)])
        touchR = valid is not None and (b >= W - 4 or not valid[vv, min(W - 1, b + 4)])
        out["main_L"][i] = np.nan if touchL else bev_px_to_xz(a, vv)[0]
        out["main_R"][i] = np.nan if touchR else bev_px_to_xz(b, vv)[0]
        others = [r for j, r in enumerate(runs) if j != mi]
        if others:
            # nearest other run (gap must be real: >= 0.5 m of non-drivable between)
            o = min(others, key=lambda r: min(abs(r[0] - b), abs(a - r[1])))
            if o[0] > b and (o[0] - b) * res >= 0.5:
                out["split_in"][i] = bev_px_to_xz(o[0], vv)[0]; out["split_out"][i] = bev_px_to_xz(o[1], vv)[0]; side[i] = 1
            elif o[1] < a and (a - o[1]) * res >= 0.5:
                out["split_in"][i] = bev_px_to_xz(o[1], vv)[0]; out["split_out"][i] = bev_px_to_xz(o[0], vv)[0]; side[i] = -1
    for k in out:
        x = out[k]
        f = np.isfinite(x)
        if f.sum() >= med:
            xs = x.copy()
            idx = np.flatnonzero(f)
            for j, ii in enumerate(idx):  # running median along z (removes row noise, keeps widening)
                lo, hi = max(0, j - med // 2), min(len(idx), j + med // 2 + 1)
                xs[ii] = np.median(x[idx[lo:hi]])
            out[k] = xs
    for k in out:  # smoothing spline along z per boundary (keeps widening, removes row-to-row jaggies)
        x = out[k]; f = np.isfinite(x)
        if f.sum() >= 6:
            sm = smooth_chain(np.stack([x[f], zs[f]], 1), zs, smooth=0.25)
            out[k] = np.where(f, sm, np.nan)
    # a split needs >= 3 m of rows; otherwise it is a notch (parked car, shadow) -> drop
    if np.isfinite(out["split_in"]).sum() * (zs[1] - zs[0]) < 3.0:
        out["split_in"][:] = np.nan; out["split_out"][:] = np.nan
    return out


def gore_polygons(stripes, lines_x=(), min_stripes=2, link_m=4.0):
    """Cluster oblique paint stripes (chevrons / hatching) into gore polygons (convex hull, metres)."""
    if len(stripes) < min_stripes:
        return []
    C = np.array([s["centre"] for s in stripes])
    n = len(C)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(C[i] - C[j])) <= link_m and abs(stripes[i]["ang"] - stripes[j]["ang"]) < np.radians(25):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    polys = []
    for g in groups.values():
        if len(g) < min_stripes:
            continue
        P = np.concatenate([stripes[i]["pts"] for i in g]).astype(np.float32)
        hull = cv2.convexHull(P).reshape(-1, 2)
        area = cv2.contourArea(hull)
        if area >= 1.0:
            polys.append({"poly": hull.astype(float), "n_stripes": len(g), "area": float(area),
                          "centre": P.mean(0).astype(float)})
    return polys


# ---------------------------------------------------------------------------------------------- tracking

class Line:
    _next = 1

    def __init__(self, x, kind, frame):
        self.id = Line._next; Line._next += 1
        self.kind = kind
        self.x = x.copy()                 # per-z state
        self.age_z = np.where(np.isfinite(x), 0, 99).astype(float)  # frames since each z was observed
        self.v_off = 0.0                  # lateral offset rate (m/frame), e.g. ego lane change
        self.sigma = 0.15                 # own innovation scale (EMA of |residual|)
        self.hits, self.miss, self.born = 1, 0, frame
        self.confirmed = False
        self.parent, self.fork_z = None, None

    def pred(self):
        return self.x + self.v_off

    def span(self):
        f = np.isfinite(self.x)
        return (ZS[f].min(), ZS[f].max()) if f.any() else (None, None)


def _cost(pred, obs, min_overlap_m=2.0, w_head=2.0):
    """Median lateral distance over the overlap + heading difference over the overlap."""
    m = np.isfinite(pred) & np.isfinite(obs)
    if m.sum() * (ZS[1] - ZS[0]) < min_overlap_m:
        return np.inf, 0
    d = obs[m] - pred[m]
    zm = ZS[m]
    head = abs(np.polyfit(zm, d, 1)[0]) if np.ptp(zm) >= 2.0 else 0.0   # relative heading (m/m)
    # divergence is allowed beyond the near overlap: use the nearest 6 m of the overlap for the offset
    near = zm <= zm.min() + 6.0
    return float(np.median(np.abs(d[near])) + w_head * head), int(m.sum())


class MultiLineTracker:
    def __init__(self, kind="lane", n_birth=3, n_death=6, alpha=0.4, gate_k=3.0, gate_min=0.3, gate_max=1.2,
                 fork_dist=1.0, z_forget=8):
        self.kind, self.n_birth, self.n_death, self.alpha = kind, n_birth, n_death, alpha
        self.gate_k, self.gate_min, self.gate_max = gate_k, gate_min, gate_max
        self.fork_dist, self.z_forget = fork_dist, z_forget
        self.lines, self.events, self.hist = [], [], {}

    def _gate(self, ln):
        return float(np.clip(self.gate_k * ln.sigma, self.gate_min, self.gate_max))

    def step(self, frame, obs_list):
        obs_list = [o for o in obs_list if np.isfinite(o).sum() >= 6]
        L = self.lines
        C = np.full((len(L), len(obs_list)), np.inf)
        for i, ln in enumerate(L):
            p = ln.pred()
            for j, o in enumerate(obs_list):
                c, _ = _cost(p, o)
                if c <= self._gate(ln):
                    C[i, j] = c
        matched_l, matched_o = set(), set()
        if C.size and np.isfinite(C).any():
            Cf = np.where(np.isfinite(C), C, 1e6)
            r, c = linear_sum_assignment(Cf)
            for i, j in zip(r, c):
                if Cf[i, j] < 1e6:
                    self._update(L[i], obs_list[j]); matched_l.add(i); matched_o.add(j)
        for i, ln in enumerate(L):
            if i not in matched_l:
                ln.miss += 1; ln.hits = max(ln.hits, 0); ln.age_z += 1
                ln.x = ln.pred()
                ln.x[ln.age_z > self.z_forget] = np.nan
        # births (+ fork / merge linking)
        for j, o in enumerate(obs_list):
            if j in matched_o:
                continue
            nl = Line(o, self.kind, frame)
            self._link_parent(nl, frame)
            self.lines.append(nl)
        alive = []
        for ln in self.lines:
            if not ln.confirmed and ln.hits >= self.n_birth:
                ln.confirmed = True
                if ln.parent is not None:
                    self.events.append({"frame": frame, "type": "fork", "child": ln.id, "parent": ln.parent, "z": ln.fork_z})
            if ln.miss > (self.n_death if ln.confirmed else 1) or not np.isfinite(ln.x).any():
                if ln.confirmed:
                    self.events.append({"frame": frame, "type": "death", "id": ln.id})
                continue
            alive.append(ln)
        self.lines = alive
        live = [self._out(ln) for ln in self.lines if ln.confirmed]
        for r in live:
            self.hist.setdefault(r["id"], {})[frame] = r
        return live

    def _update(self, ln, o):
        p = ln.pred()
        m = np.isfinite(o)
        both = m & np.isfinite(p)
        if both.any():
            res = o[both] - p[both]
            near = ZS[both] <= ZS[both].min() + 6.0
            r0 = float(np.median(res[near]))
            ln.v_off = 0.7 * ln.v_off + 0.3 * float(np.clip(r0 + ln.v_off, -0.3, 0.3))
            ln.sigma = 0.8 * ln.sigma + 0.2 * float(min(np.median(np.abs(res)), 1.0))
        new = p.copy()
        new[both] = (1 - self.alpha) * p[both] + self.alpha * o[both]
        only_o = m & ~np.isfinite(p)
        new[only_o] = o[only_o]                       # span grows where newly observed (divergence kept)
        ln.age_z[m] = 0
        ln.age_z[~m] += 1
        new[(ln.age_z > self.z_forget)] = np.nan
        f_ = np.isfinite(new)
        if f_.sum() >= 8:  # iter5: re-smooth the blended curve (EMA alone leaves per-sample wiggle -> jit2)
            new = np.where(f_, smooth_chain(np.stack([new[f_], ZS[f_]], 1), ZS, smooth=0.5), np.nan)
        ln.x = new
        ln.hits += 1; ln.miss = 0

    def _link_parent(self, nl, frame):
        zlo, zhi = nl.span()
        if zlo is None:
            return
        for ln in self.lines:
            if not ln.confirmed:
                continue
            for zend in (zlo, zhi):
                k = int(round((zend - ZS[0]) / (ZS[1] - ZS[0])))
                if 0 <= k < len(ZS) and np.isfinite(ln.x[k]) and abs(ln.x[k] - nl.x[k]) < self.fork_dist:
                    nl.parent, nl.fork_z = ln.id, float(zend)
                    return
        # converging / diverging lines: two lines closer than fork_dist anywhere in the overlap
        for ln in self.lines:
            if not ln.confirmed:
                continue
            m = np.isfinite(ln.x) & np.isfinite(nl.x)
            if m.sum() >= 4:
                d = np.abs(ln.x[m] - nl.x[m])
                k = int(np.argmin(d))
                if d[k] < self.fork_dist and d.max() > 2 * self.fork_dist:
                    nl.parent, nl.fork_z = ln.id, float(ZS[m][k])
                    return

    def _out(self, ln):
        return {"id": ln.id, "kind": ln.kind, "x": ln.x.copy(), "parent": ln.parent, "fork_z": ln.fork_z,
                "hits": ln.hits, "held": ln.miss > 0, "sigma": ln.sigma}


class GoreTracker:
    def __init__(self, n_birth=2, n_death=4, link_m=3.0, alpha=0.5):
        self.n_birth, self.n_death, self.link_m, self.alpha = n_birth, n_death, link_m, alpha
        self.g = []

    def step(self, polys):
        used = set()
        for t in self.g:
            best, bd = None, None
            for j, p in enumerate(polys):
                if j in used:
                    continue
                d = np.hypot(*(p["centre"] - t["centre"]))
                if d < self.link_m and (bd is None or d < bd):
                    best, bd = j, d
            if best is None:
                t["miss"] += 1
            else:
                used.add(best); p = polys[best]
                t["centre"] = (1 - self.alpha) * t["centre"] + self.alpha * p["centre"]
                t["poly"] = p["poly"]; t["hits"] += 1; t["miss"] = 0
        for j, p in enumerate(polys):
            if j not in used:
                self.g.append(dict(p, hits=1, miss=0))
        self.g = [t for t in self.g if t["miss"] <= self.n_death]
        return [t for t in self.g if t["hits"] >= self.n_birth]


def curve_points(x, zs=ZS):
    f = np.isfinite(x)
    return np.stack([x[f], zs[f]], 1)


# ---------------------------------------------------------------------------------------------- iter4 stage 2
def _legacy(rec, side=None, kind="lane", extra=None):
    """lines_mt record -> foundation_bev line record (poly2 'c' kept for legacy consumers; drawing and
    the metrics use 'xg' through bev_smooth.sample_curve)."""
    x = rec["x"]; f = np.isfinite(x)
    z = ZS[f]; xx = x[f]
    c = np.polyfit(z, xx, 2 if np.ptp(z) > 12 else 1)[::-1] if f.sum() >= 3 else np.array([xx.mean() if f.any() else 0.0])
    c = np.concatenate([c, np.zeros(3 - len(c))])
    if side is None:
        side = -1 if np.median(xx[z <= z.min() + 6]) < 0 else 1
    out = {"id": int(rec["id"]), "side": int(side), "c": c, "zmin": float(z.min()), "zmax": float(z.max()),
           "hits": int(rec["hits"]), "held": bool(rec["held"]), "xg": x, "parent": rec.get("parent"),
           "fork_z": rec.get("fork_z"), "kind": kind}
    if extra:
        out.update(extra)
    return out


def _plausible(x, max_slope=0.5, max_step=0.3):
    """Reject near-horizontal / zigzag curves: |dx/dz| median and row-to-row lateral step."""
    f = np.isfinite(x)
    if f.sum() < 6:
        return False
    d = np.diff(x[f]) / np.diff(ZS[f])
    return float(np.median(np.abs(d))) <= max_slope and float(np.percentile(np.abs(np.diff(x[f])), 90)) <= max_step


def clean_lanes(lanes, min_len=6.0, min_hits=4, per_side=4, z_ref=10.0):
    """iter5 clutter control: min length + age, no near-horizontal/zigzag, no crossings, <= per_side per side."""
    ok = []
    for r in lanes:
        f = np.isfinite(r["xg"])
        # iter6: reject only if BOTH short and young (iter5 required both, killing real lanes)
        if f.sum() < 2 or ((ZS[f].max() - ZS[f].min()) < min_len and r["hits"] < min_hits):
            continue
        if not _plausible(r["xg"], max_slope=0.5, max_step=0.5):  # iter6: zigzag step 0.3 -> 0.5 m
            continue
        ok.append(r)
    ok.sort(key=lambda r: -r["hits"])
    keep = []
    for r in ok:  # crossing: lateral difference changes sign by > 0.3 m over the common span -> keep the older
        cross = False
        for q in keep:
            both = np.isfinite(r["xg"]) & np.isfinite(q["xg"])
            if both.sum() >= 4:
                d = r["xg"][both] - q["xg"][both]
                if d.max() > 0.3 and d.min() < -0.3:
                    cross = True; break
                if np.abs(d).max() < 0.6 or (np.abs(d).min() < 0.4 and r.get("parent") != q["id"] and q.get("parent") != r["id"]):
                    cross = True; break  # duplicate on the same paint
        if not cross:
            keep.append(r)
    out = []
    for sd in (-1, 1):
        ss = [r for r in keep if r["side"] == sd]
        ss.sort(key=lambda r: abs(np.interp(z_ref, ZS[np.isfinite(r["xg"])], r["xg"][np.isfinite(r["xg"])])))
        out += ss[:per_side]
    return out


def _near_angle(x, z_near=(5.0, 15.0)):
    f = np.isfinite(x) & (ZS >= z_near[0]) & (ZS <= z_near[1])
    if f.sum() < 4:
        f = np.isfinite(x)
        f &= ZS <= (ZS[f].min() + 10.0) if f.any() else f
    if f.sum() < 4:
        return None
    return float(np.degrees(np.arctan(np.polyfit(ZS[f], x[f], 1)[0])))


def heading_ok(x, max_deg=float(os.environ.get("FBEV_EDGE_MAX_DEG", 15.0)), lanes=()):
    """iter6: an edge must run within max_deg of the road heading near the ego. Road heading = median heading of
    the tracked lanes (so curves / turns, e.g. munich, are not rejected), else the ego heading (0)."""
    a = _near_angle(x)
    if a is None:
        return True
    la = [v for v in (_near_angle(r["xg"]) for r in lanes) if v is not None]
    ref = float(np.median(la)) if la else 0.0
    return abs(a - ref) <= max_deg


def crosses_lane(x, lanes, tol=0.5, min_n=6):
    """iter6: True if the curve x(ZS) passes from one side of a tracked lane line to the other (> tol each way)."""
    for r in lanes:
        both = np.isfinite(x) & np.isfinite(r["xg"])
        if both.sum() >= 6:
            d = x[both] - r["xg"][both]
            if (d > tol).sum() >= min_n and (d < -tol).sum() >= min_n:  # >= 3 m clearly on each side
                return True
    return False


def xg_of(r):
    """x on ZS for a line record (lines_mt 'xg', else the legacy poly 'c' over zmin..zmax)."""
    if r.get("xg") is not None:
        return np.asarray(r["xg"], float)
    c = np.asarray(r.get("c", [0, 0, 0]), float)
    x = c[0] + c[1] * ZS + (c[2] * ZS ** 2 if len(c) > 2 else 0.0)
    return np.where((ZS >= r.get("zmin", 0)) & (ZS <= r.get("zmax", 0)), x, np.nan)


def _longest_run(mask, min_n):
    best, cur, bs = (0, 0), 0, 0
    for i, m in enumerate(mask):
        if m:
            if cur == 0:
                bs = i
            cur += 1
            if cur > best[1] - best[0]:
                best = (bs, i + 1)
        else:
            cur = 0
    return best if best[1] - best[0] >= min_n else None


EDGE_CONV_DEG = float(os.environ.get("FBEV_EDGE_MAX_DEG", 30.0))      # iter8: converging (toward the lanes) limit
EDGE_DIV_DEG = float(os.environ.get("FBEV_EDGE_DIVERGE_DEG", 45.0))   # iter8: diverging (away from the road) limit; 60 -> munich R 94.1 but keeps 45-60 deg corner flares
EGO_HALF_M = float(os.environ.get("FBEV_EDGE_EGO_HALF_M", 1.2))       # iter8: an edge never enters the ego corridor


_TRIM_DEBUG_PATH = os.environ.get("FBEV_TRIM_DEBUG", "")
EDGE_GAP_M = float(os.environ.get("FBEV_EDGE_GAP_M", "5.0"))        # iter10: bridge interior holes <= this before trim
EDGE_GAP_SLOPE = float(os.environ.get("FBEV_EDGE_GAP_SLOPE", "0.5"))  # iter10: |bridge slope - road slope| limit


def bridge_gaps(x, max_gap_m=None, max_slope=None, ref_slope=0.0):
    """iter10 (munich R: 40 frames lost z = 10 m with a raw edge present): the SAM road boundary has interior holes
    (parked / stopped cars, shadows) of 1-6.5 m; trim_edge keeps the NEAREST run >= 4 m, so a 1.5 m near piece + a
    hole made it skip to the far part (f2: 21-44 m, f258: 11.5-15 m). Interior holes <= max_gap_m are filled
    linearly when the bridge runs with the road (|slope - ref_slope| <= max_slope = 0.5, 27 deg: 1.0 let the forecourt hooks f8-12 / f144-147 back in); the trim rules (lane crossing,
    ego corridor, heading) then still judge the bridged samples, so flares / crossings stay out. Leading /
    trailing NaN are never extrapolated; longer holes (f2-7: 6.5 m in front of the parked cars made a diagonal hook; forecourts: munich f158-165 10.5 m,
    f256-265 11-19 m) stay holes."""
    max_gap_m = EDGE_GAP_M if max_gap_m is None else max_gap_m
    max_slope = EDGE_GAP_SLOPE if max_slope is None else max_slope
    x = np.asarray(x, float).copy()
    idx = np.flatnonzero(np.isfinite(x))
    if max_gap_m <= 0 or len(idx) < 2:
        return x
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a < 2:
            continue
        dz = ZS[b] - ZS[a]
        if dz > max_gap_m + 1e-6 or abs((x[b] - x[a]) / dz - ref_slope) > max_slope:
            continue
        x[a + 1:b] = np.interp(ZS[a + 1:b], [ZS[a], ZS[b]], [x[a], x[b]])
    return x


TRIM_DEBUG = None  # list -> trim_edge appends its masks (offline diagnosis only)


def trim_edge(x, side, lanes, max_deg=None, z_near=20.0, tol=0.5, min_len_m=4.0, div_deg=None, ego_half=None):
    """iter7: instead of dropping a whole edge, remove only the bad samples. iter8 rules:
    (a) lane crossing: an edge that starts outside a tracked lane line and passes > tol to its inner side is
        invalid from the first crossing sample ON (iter7 kept the longest run, which on lane_split f23-29 was the
        far diagonal part through the ego lane, and dropped the good near part);
    (b) ego corridor: samples with side*x < ego_half are bad (an edge cannot run through the ego lane);
    (c) heading, near ego (z <= z_near), split by direction w.r.t. the road heading: CONVERGING with distance
        (toward the lanes, the lane_split / munich-L diagonal) > max_deg (30) is bad; DIVERGING (away from the
        road: junction, driveway, kerb flare; 199 munich-R frames at 30-55 deg, none crossing a lane) only > div_deg (45);
    then keep the NEAREST remaining run >= min_len_m (None if there is none)."""
    max_deg = EDGE_CONV_DEG if max_deg is None else max_deg
    div_deg = EDGE_DIV_DEG if div_deg is None else div_deg
    ego_half = EGO_HALF_M if ego_half is None else ego_half
    x = np.asarray(x, float).copy()
    la = [v for v in (_near_angle(xg_of(r)) for r in lanes) if v is not None]
    ref = np.radians(float(np.median(la))) if la else 0.0
    x = bridge_gaps(x, ref_slope=float(np.tan(ref)))  # iter10: munich R f2-12 / 144-173 / 256-267 (occlusion holes)
    f = np.isfinite(x)
    if f.sum() < 4:
        return None
    bad = np.zeros(len(ZS), bool)
    for r in lanes:
        lx = xg_of(r)
        both = f & np.isfinite(lx)
        idx = np.flatnonzero(both)
        if len(idx) < 6:
            continue
        # only lanes that start INSIDE the edge (between ego and edge at the near end) can be crossed; lines lying
        # beyond the edge (parking / kerb paint, munich and a2d2) must not trim it
        if np.median((x[idx[:4]] - lx[idx[:4]]) * side) <= 0:
            continue
        cross = both & ((x - lx) * side < -tol)
        ci = np.flatnonzero(cross)
        if len(ci) >= 2:  # 2 samples (1 m) to ignore a single noisy sample
            bad[ci[0]:] = True
        else:
            bad |= cross
    bad |= f & (x * side < ego_half)
    xs = np.where(f, x, np.interp(ZS, ZS[f], x[f]))
    dev = np.degrees(np.arctan(np.gradient(xs, ZS)) - ref) * side  # > 0: diverging (moving outward with distance)
    near = f & (ZS <= z_near)
    bad |= near & (((dev < 0) & (-dev > max_deg)) | ((dev > 0) & (dev > div_deg)))
    run = _nearest_run(f & ~bad, int(round(min_len_m / (ZS[1] - ZS[0]))))
    if TRIM_DEBUG is not None:
        TRIM_DEBUG.append({"side": side, "x": x.copy(), "bad": bad.copy(), "run": run, "ref": float(np.degrees(ref)),
                           "dev": dev.copy(), "lanes": [np.asarray(xg_of(r), float).copy() for r in lanes]})
    if run is None:
        return None
    out = np.full(len(ZS), np.nan)
    out[run[0]:run[1]] = x[run[0]:run[1]]
    return out


def is_flared(x, side, lanes, max_deg=None, z_near=20.0):
    """iter8: True if the kept edge uses the diverging allowance (near-ego heading > max_deg AWAY from the road).
    Such a main edge must not seed a split_in-vs-main fork gore (re5 munich f587-593 fired once flares were kept)."""
    max_deg = EDGE_CONV_DEG if max_deg is None else max_deg
    if x is None:
        return False
    x = np.asarray(x, float); f = np.isfinite(x)
    if f.sum() < 4:
        return False
    la = [v for v in (_near_angle(xg_of(r)) for r in lanes) if v is not None]
    ref = np.radians(float(np.median(la))) if la else 0.0
    xs = np.where(f, x, np.interp(ZS, ZS[f], x[f]))
    dev = np.degrees(np.arctan(np.gradient(xs, ZS)) - ref) * side
    return bool((f & (ZS <= z_near) & (dev > max_deg)).any())


def _nearest_run(mask, min_n):
    """iter8: first (nearest-to-ego) contiguous True run of at least min_n samples, as (start, stop)."""
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_n:
                return i, j
            i = j
        else:
            i += 1
    return None


def suppress_edge_duplicates(lanes, edges, d_max=None, frac=0.7, min_n=12):
    """iter7: drop a lane line that runs within d_max of a road edge over >= frac of their common span
    (exit_approach f0020: right-most lane 0.3 m from the right edge duplicated it)."""
    # GPU exit_approach f0018-20: gap 0.2 m near ego -> 0.7 m at 35 m (painted edge line vs asphalt edge), so 0.8 m
    d_max = float(os.environ.get("FBEV_LANE_EDGE_DUP_M", "0.8")) if d_max is None else d_max
    E = [xg_of(e) for e in edges]
    keep = []
    near = {}
    for r in lanes:  # the nearest line on each side is never suppressed (keeps the nearest-L/R lane metric)
        lx = xg_of(r); fx = np.isfinite(lx)
        if fx.any():
            k10 = np.flatnonzero(fx)[np.argmin(np.abs(ZS[fx] - 10.0))]  # the metric's nearest-L/R is taken at z=10
            sd = -1 if lx[k10] < 0 else 1
            d = abs(lx[k10])
            if sd not in near or d < near[sd][0]:
                near[sd] = (d, id(r))
            i10 = int(np.argmin(np.abs(ZS - 10.0)))
            if fx[i10] and ((sd, 10) not in near or d < near[(sd, 10)][0]):  # nearest among lines that reach z=10
                near[(sd, 10)] = (d, id(r))
    nearest = {v[1] for v in near.values()}
    for r in lanes:
        lx = xg_of(r); dup = False
        if id(r) in nearest:
            keep.append(r); continue
        for ex in E:
            both = np.isfinite(lx) & np.isfinite(ex)
            if both.sum() >= min_n and (np.abs(lx[both] - ex[both]) <= d_max).mean() >= frac:
                dup = True; break
        if not dup:
            keep.append(r)
    return keep


EDGE_FRAG_M = float(os.environ.get("FBEV_EDGE_FRAG_M", "6.0"))  # iter11: a live edge shorter than this is a fragment
EDGE_FRAG_EXT_M = float(os.environ.get("FBEV_EDGE_FRAG_EXT_M", "12.0"))  # iter11: max near extension of a fragment
EDGE_FRAG_DEV = float(os.environ.get("FBEV_EDGE_FRAG_DEV", "0.25"))     # iter11: fragment slope vs lane heading veto
# iter13 (GPU iter12 munich R f562-563, f570-571): the veto also applies to a fragment the BEV edge tracker coasts
# (held, no EdgeHold hold_age); iter11-12 let a coasted forecourt hook through because it carried held=True
EDGE_FRAG_COAST = os.environ.get("FBEV_EDGE_FRAG_COAST", "1") == "1"


def is_diag_fragment(e, lanes, z_near=10.0, max_dev=None):
    """iter11 (GPU iter10 munich R f520-526: z 20.5-24.5 m, slope 0.37; f557-560: z 11.5-17 m, slope 0.94; lane
    heading 0.03-0.07): gap bridging + trim left a short diagonal piece of the image-border / forecourt hook as the
    live main edge, which replaced the held line iter9 coasted. A live main edge shorter than EDGE_FRAG_M that starts
    beyond z_near and deviates from the lane heading by > max_dev is not a kerb: drop it for this frame."""
    max_dev = EDGE_FRAG_DEV if max_dev is None else max_dev
    if EDGE_FRAG_M <= 0 or e is None or e.get("kerb_occ"):
        return False
    if e.get("held") and (not EDGE_FRAG_COAST or "hold_age" in e):
        return False
    x = np.asarray(e["xg"], float); f = np.isfinite(x)
    if f.sum() < 4 or np.ptp(ZS[f]) >= EDGE_FRAG_M or ZS[f].min() <= z_near:
        return False
    return abs(float(np.polyfit(ZS[f], x[f], 1)[0]) - _ref_slope(None, lanes)) > max_dev


def frag_extend(h, side, lanes, z_near=10.0):
    """iter11 (GPU iter10 munich R f520-526, f557-560): gap bridging turned a fully-trimmed frame (iter9: no live
    edge -> the hold coasted the whole line) into a short far fragment (z 12-25 m) that replaced the held line.
    At z = 10 m these kerbs are at or beyond the right image border (munich: u >= 1220 of 1280), so iter9's 'presence'
    was a coasted hook (x10 4.6 -> 6.8 m in 7 frames). Instead of the hook, a fragment (span < EDGE_FRAG_M, not
    reaching z_near, starting by z_near + EDGE_FRAG_EXT_M) that itself runs along the road heading (|slope - lane
    heading| <= 0.15) is extended to z_near along that heading, like the clip-start lane backfill; the extension
    must pass trim_edge (no lane crossing / ego corridor)."""
    if EDGE_FRAG_M <= 0 or h is None or h.get("held"):
        return h
    cx = np.asarray(h["xg"], float)
    f = np.isfinite(cx)
    if f.sum() < 4 or np.ptp(ZS[f]) >= EDGE_FRAG_M or ZS[f].min() <= z_near or ZS[f].min() > z_near + EDGE_FRAG_EXT_M:
        return h
    ref = _ref_slope(None, lanes)
    sl = float(np.polyfit(ZS[f], cx[f], 1)[0])
    if abs(sl - ref) > 0.15:
        return h
    z0 = ZS[f].min(); x0 = float(np.median(cx[f][:3]))
    m = (ZS >= z_near) & (ZS < z0)
    xo = cx.copy(); xo[m] = x0 + ref * (ZS[m] - z0)
    xt = trim_edge(xo, side, lanes)
    i0 = int(np.argmin(np.abs(ZS - z_near)))
    if xt is None or not np.isfinite(xt[i0]) or (f & ~np.isfinite(xt)).any():
        return h
    fz = ZS[np.isfinite(xt)]
    return dict(h, xg=xt, zmin=float(fz.min()), zmax=float(fz.max()), held=False, frag_override=True)


def retrim_held(h, cur, side, lanes):
    """iter8: held / filled geometry must pass trim_edge too, and a fill may only ADD samples: if re-trimming the
    filled edge loses any sample the current (trimmed) edge measured, the current edge is used unfilled
    (city f23: an old near-ego fill crossed a lane and, with crossing-invalidates-beyond, cost the whole edge)."""
    if h is None or not (h.get("hold_filled") or h.get("held")):
        return h
    xt = trim_edge(h["xg"], side, lanes)
    if cur is not None:
        cx = xg_of(cur)
        if xt is None or (np.isfinite(cx) & ~np.isfinite(xt)).any():
            return cur
    if xt is None:
        return None
    fz = ZS[np.isfinite(xt)]
    return dict(h, xg=xt, zmin=float(fz.min()), zmax=float(fz.max()))


EDGE_ANCHOR_S = float(os.environ.get("FBEV_EDGE_ANCHOR_S", "3.0"))     # iter12: lane-anchored coast time (0 = off)
EDGE_ANCHOR_DEV = float(os.environ.get("FBEV_EDGE_ANCHOR_DEV", "0.15"))  # iter12: learn only from lane-parallel edges
# iter13 (GPU iter12 munich R f190-196, f215, f240, f250: x10 8.6 -> 6.8 -> 9.8 m when the same-side lane drops out and
# the reference flips to the opposite edge, whose offset was learned at another time): an anchored line moves at z_ref
# by at most EDGE_ANCHOR_RATE m/s from the previous frame's output of that role (0 = off)
EDGE_ANCHOR_RATE = float(os.environ.get("FBEV_EDGE_ANCHOR_RATE", "1.0"))


def _nearest_lane_x(lanes, side, z_ref=10.0):
    """nearest same-side tracked lane at z_ref (no fallback): its x samples, or None."""
    iz = int(np.argmin(np.abs(ZS - z_ref)))
    best = None
    for r in lanes or []:
        if r.get("fallback"):
            continue
        x = xg_of(r)
        if x is None:
            continue
        x = np.asarray(x, float)
        if not np.isfinite(x[iz]) or side * x[iz] <= 0:
            continue
        if best is None or side * x[iz] < side * best[iz]:
            best = x
    return best


class LaneAnchor:
    """iter12 (GPU iter11 munich R f561-577, f165-166): when the main edge has no valid measurement (the raw edge runs
    along a moving car's bottom and the far piece is a forecourt hook, so trim + the fragment veto leave nothing and the
    0.5 s travel-shift hold expires, or the hold carries a diagonal hook outward by 0.25 m / frame), a road edge still
    runs parallel to the lanes and to the opposite road edge. Learn, per role, the lateral offset at z_ref between a
    LIVE main edge (kerb-pushed allowed: the kerb line follows the road heading) that runs parallel to the reference
    (|slope difference| <= EDGE_ANCHOR_DEV over z <= 25 m) and (a) the nearest same-side lane, (b) the opposite live
    main edge. For up to EDGE_ANCHOR_S after that, a missing or EdgeHold-coasted edge is drawn as reference + offset
    (a first, b if no same-side lane is live), if trim_edge accepts it. Marked held + anchored; anchored lines are
    never learned from."""

    def __init__(self, fps=10.0, z_ref=10.0):
        self.max_age = int(round(EDGE_ANCHOR_S * fps)); self.z_ref = z_ref
        self.iz = int(np.argmin(np.abs(ZS - z_ref)))
        self.s = {}
        self.step_m = EDGE_ANCHOR_RATE / max(float(fps), 1e-6)  # iter13: max |dx| at z_ref per frame
        self.prev = {}  # iter13: role -> x at z_ref of the previous frame's final output (None: none there)

    def note(self, role, e):
        """iter13: remember the final output of this frame (after the car clearance) for the continuity limit."""
        x = None if e is None or e.get("xg") is None else float(np.asarray(e["xg"], float)[self.iz])
        self.prev[role] = x if x is not None and np.isfinite(x) else None

    def _limit(self, role, x):
        """iter13: rate-limit a new anchored line at z_ref against the previous output (a parallel shift)."""
        p = self.prev.get(role)
        if self.step_m <= 0 or p is None or not np.isfinite(x[self.iz]):
            return x
        d = float(x[self.iz]) - p
        if abs(d) <= self.step_m:
            return x
        return x - (d - np.sign(d) * self.step_m)

    def _off(self, x, ref, side):
        if ref is None:
            return None
        m = np.isfinite(x) & np.isfinite(ref) & (ZS <= 25.0)
        if m.sum() < 8 or not np.isfinite(ref[self.iz]):
            return None
        if abs(float(np.polyfit(ZS[m], x[m] - ref[m], 1)[0])) > EDGE_ANCHOR_DEV:
            return None
        return float(x[self.iz] - ref[self.iz])

    def learn(self, role, e, side, lanes, other=None):
        for kk in list(self.s.get(role, {})):
            st = self.s[role][kk]; st["age"] += 1
            if st["age"] > self.max_age:
                self.s[role].pop(kk)
        if self.max_age < 1 or e is None or e.get("held") or e.get("backfill") or e.get("anchored"):
            return
        x = np.asarray(e["xg"], float)
        if not np.isfinite(x[self.iz]):
            return
        ln = _nearest_lane_x(lanes, side, self.z_ref)
        o = self._off(x, ln, side)
        if o is not None and side * o >= 0.5:
            self.s.setdefault(role, {})["lane"] = {"off": o, "age": 0}
        ox = None if other is None or other.get("held") else np.asarray(other["xg"], float)
        o = self._off(x, ox, side)
        if o is not None and side * o >= 2.5:
            self.s.setdefault(role, {})["edge"] = {"off": o, "age": 0}

    def fill(self, role, h, side, lanes, other=None):
        if self.max_age < 1:
            return h
        if h is not None and not (h.get("held") and "hold_age" in h):
            # live edges stay (a near gap may be filled). So does an edge the BEV tracker coasts (held, no hold_age):
            # only EdgeHold's travel-shift coasts (hold_age set) and missing edges are replaced. Replacing tracker
            # coasts too raised the CPU-smoke edge jit2 (lane_split L 0.025 -> 0.081) and is invisible to the replay.
            return self._extend(role, h, side, lanes, other)
        if h is not None and h.get("kerb_occ") and np.isfinite(np.asarray(h["xg"], float)[self.iz]):
            return h  # a held kerb line (iter9 kerb hold) that still reaches z_ref stays
        x = self._line(role, side, lanes, other)
        if x is None:
            return h
        x = self._limit(role, x)
        xt = trim_edge(x, side, lanes)
        if xt is None or not np.isfinite(xt[self.iz]):
            return h
        fz = ZS[np.isfinite(xt)]
        base = h if h is not None else {"id": 0, "hits": 0, "role": role, "side": side, "kind": "edge", "split": False}
        return dict(base, xg=xt, zmin=float(fz.min()), zmax=float(fz.max()), held=True, anchored=True,
                    kerb_occ=False, kerb_alpha=0.0, c=np.zeros(3))

    def _extend(self, role, h, side, lanes, other=None, join_m=0.6):
        """a live edge that starts beyond z_ref (munich f572-575: z 12.5-20 m) gets the anchored line below its near
        end, if the two meet within join_m there; samples are only added, and the result must pass trim_edge."""
        cx = np.asarray(h["xg"], float); f = np.isfinite(cx)
        if f.sum() < 4 or np.isfinite(cx[self.iz]) or h.get("kerb_occ"):
            return h
        x = self._line(role, side, lanes, other)
        if x is None:
            return h
        i0 = int(np.flatnonzero(f)[0])
        if not np.isfinite(x[i0]) or abs(x[i0] - cx[i0]) > join_m:
            return h
        xo = cx.copy(); m = (np.arange(len(ZS)) < i0) & np.isfinite(x) & (ZS >= self.z_ref - 1e-6)
        xo[m] = x[m] + (cx[i0] - x[i0])  # shifted to meet the live edge
        xt = trim_edge(xo, side, lanes)
        if xt is None or not np.isfinite(xt[self.iz]) or (f & ~np.isfinite(xt)).any():
            return h
        fz = ZS[np.isfinite(xt)]
        return dict(h, xg=xt, zmin=float(fz.min()), zmax=float(fz.max()), anchored=True)

    def _line(self, role, side, lanes, other=None):
        st = self.s.get(role) or {}
        ref = None
        if "lane" in st:
            ln = _nearest_lane_x(lanes, side, self.z_ref)
            if ln is not None:
                ref, off = ln, st["lane"]["off"]
        if ref is None and "edge" in st and other is not None and not other.get("held"):
            ref, off = np.asarray(other["xg"], float), st["edge"]["off"]
        if ref is None:
            return None
        x = np.where(np.isfinite(ref), ref + off, np.nan)
        x[ZS > 40.0] = np.nan
        return x

class EdgeHold:
    """iter7: temporal hold per edge role. The last accepted edge is shifted by the ego travel ds
    (x_new(z) = x_old(z + ds)) and kept per sample for up to max_age frames, with confidence decay**age.
    - edge missing this frame: the propagated edge is emitted (held=True);
    - edge present but trimmed / short: its missing samples are filled from the propagated edge, so the
      near-ego part removed by trim_edge is bridged by the last good geometry instead of disappearing."""

    def __init__(self, max_age=None, decay=0.8, fps=10.0):
        # iter8: hold for a fixed TIME (FBEV_EDGE_HOLD_S, 0.5 s), not 5 frames: 5 frames is 0.5 s at 10 Hz but only
        # 0.17 s on munich (30 Hz), where 169 held frames still left 246 frames without a right edge.
        hold_s = float(os.environ.get("FBEV_EDGE_HOLD_S", 0.5))
        self.max_age = max_age if max_age is not None else max(fr10(5, fps, 2), int(round(hold_s * fps)))
        self.decay = decay ** (5.0 / self.max_age)  # same confidence at the end of the hold
        self.last = {}

    @staticmethod
    def _shift(x, ds):
        f = np.isfinite(x)
        out = np.full(len(ZS), np.nan)
        if f.sum() >= 2:
            zsrc = ZS + ds
            m = (zsrc >= ZS[f].min()) & (zsrc <= ZS[f].max())
            out[m] = np.interp(zsrc[m], ZS[f], x[f])
        return out

    def step(self, role, rec, ds):
        prev = self.last.get(role)
        px = pa = None
        if prev is not None:
            r0, x0, a0 = prev
            px = self._shift(x0, ds)
            pa = np.round(self._shift(a0.astype(float), ds)) + 1
            pa[~np.isfinite(px)] = np.inf
            px[pa > self.max_age] = np.nan
        if rec is not None:
            x = np.asarray(rec["xg"], float).copy()
            age = np.where(np.isfinite(x), 0.0, np.inf)
            if px is not None:
                fill = ~np.isfinite(x) & np.isfinite(px)
                x[fill] = px[fill]; age[fill] = pa[fill]
            out = dict(rec, xg=x)
            if np.isfinite(age[np.isfinite(x)]).any() and (age[np.isfinite(x)] > 0).any():
                out["hold_filled"] = int((age[np.isfinite(x)] > 0).sum())
            self.last[role] = (out, x, np.where(np.isfinite(x), age, np.inf))
            fz = ZS[np.isfinite(x)]; out["zmin"], out["zmax"] = float(fz.min()), float(fz.max())
            return out
        if px is None or np.isfinite(px).sum() < 4:
            self.last.pop(role, None)
            return None
        fa = pa[np.isfinite(px)]
        out = dict(prev[0], xg=px, held=True, hold_age=int(fa.min()),
                   hold_conf=round(float(self.decay ** fa.min()), 3))
        fz = ZS[np.isfinite(px)]; out["zmin"], out["zmax"] = float(fz.min()), float(fz.max())
        self.last[role] = (out, px, np.where(np.isfinite(px), pa, np.inf))
        return out


GORE_KERB_RESET = os.environ.get("FBEV_GORE_KERB_RESET", "1") == "1"  # iter11


class DivergenceGore:
    """iter6: persistent divergence gores. A candidate (per-frame fork_gores wedge, or a TEMPORAL widening of the
    gap between the outermost lane line and that side's main edge) must persist >= n_persist consecutive frames
    with a non-decreasing gap before it is drawn."""

    def __init__(self, n_persist=3, grow_m=float(os.environ.get("FBEV_GORE_GROW_M", 1.2)), hist=12,
                 max_step=float(os.environ.get("FBEV_GORE_MAX_STEP", 0.4))):
        self.n_persist, self.grow_m, self.hist, self.max_step = n_persist, grow_m, hist, max_step
        self.run = {}      # key -> consecutive frames
        self.gaps = {-1: [], 1: []}
        self.kalpha = {-1: 0.0, 1: 0.0}

    def step(self, lanes, edges):
        cands = {}
        # iter6: the single-frame lane-vs-edge wedge fired persistently on urban kerbs (munich, a2d2, city);
        # the temporal "widen" rule below replaces it. Child-lane forks and split edges are kept.
        for g in fork_gores(lanes, edges, lane_edge=False):
            cands[("fork", int(np.sign(g["centre"][0])))] = g
        mains = {e["side"]: e for e in edges if e.get("role") in ("main_L", "main_R")}
        for sd in (-1, 1):
            ls = [r for r in lanes if r["side"] == sd and np.isfinite(r["xg"]).sum() >= 8]
            e = mains.get(sd)
            if not ls or e is None:
                self.gaps[sd] = []
                continue
            # iter6: the through-lane line nearest the ego (GPU exit_approach: the exit lane's own outer line hugs the
            # SAM edge, so outermost-lane -> edge never widens; nearest-line -> edge grows 2.2 -> 4.0 m over f8-f20)
            nl = min(ls, key=lambda r: abs(np.nanmedian(r["xg"])))
            both = np.isfinite(nl["xg"]) & np.isfinite(e["xg"]) & (ZS >= 6.0) & (ZS <= 30.0)
            if both.sum() < 8:
                self.gaps[sd] = []
                continue
            gap = (e["xg"] - nl["xg"]) * sd
            gm = float(np.median(gap[both & (ZS <= 20.0)])) if (both & (ZS <= 20.0)).sum() >= 4 else float(np.median(gap[both]))
            H = (self.gaps[sd] + [gm])[-self.hist:]
            if len(H) >= 2 and abs(H[-1] - H[-2]) > self.max_step:
                H = [gm]  # a jump (lane change, edge flip) restarts the history: growth must be gradual
            # iter11 (GPU iter10 munich f881-885 false widen gore): the iter10 kerb blend moves the main edge outward
            # by up to KERB_MAX_PUSH_M in 1 / ramp_n steps (0.25-0.37 m / frame on munich, under max_step), so a kerb
            # re-interpretation looked like a gradually opening exit lane. The iter9 switch jumped and restarted the
            # history. A change of the kerb blend weight restarts the history too: the edge's model changed, the
            # road did not.
            ka = float(e.get("kerb_alpha", 1.0 if e.get("kerb_occ") else 0.0))
            if GORE_KERB_RESET and abs(ka - self.kalpha.get(sd, 0.0)) > 1e-6:
                H = [gm]
            self.kalpha[sd] = ka
            self.gaps[sd] = H
            base = min(H)
            steps = np.diff(H[-4:]) if len(H) >= 4 else np.array([np.inf])
            # growing (>= grow_m above the recent minimum), monotone over the last 3 steps, and smooth
            # (each step <= max_step: a lane change or an edge flip jumps, an exit lane opens gradually)
            if len(H) >= 4 and gm - base >= self.grow_m and (steps >= -0.05).all() and (steps <= self.max_step).all():
                sel = both & (gap - base >= 0.3)
                if sel.sum() >= 4:
                    xa = nl["xg"][sel] + sd * base; xb = e["xg"][sel]; zz = ZS[sel]
                    poly = np.concatenate([np.stack([xa, zz], 1), np.stack([xb, zz], 1)[::-1]])
                    cands[("widen", sd)] = {"poly": poly, "n_stripes": 0, "area": float(_trapz(np.abs(xb - xa), zz)),
                                            "centre": poly.mean(0), "src": "widen"}
        out = []
        for k in list(self.run):
            if k not in cands:
                self.run[k] = 0
        for k, g in cands.items():
            self.run[k] = self.run.get(k, 0) + 1
            # spatial (single-frame shape) wedges flicker on city clutter -> 2x persistence; temporal widening 1x
            if self.run[k] >= (self.n_persist if k[0] == "widen" else 2 * self.n_persist):
                out.append(g)
        return out


def fork_gores(lanes, edges, min_gap=1.0, max_gap=4.0, lane_edge=True):
    """Gore between a diverging pair (child lane vs parent lane, or split edge vs main edge on its side):
    the gap must grow with z from < min_gap near the fork to >= min_gap; polygon spans the diverging part."""
    pairs = []
    by_id = {r["id"]: r for r in lanes}
    for r in lanes:
        if r.get("parent") in by_id:
            pairs.append((by_id[r["parent"]]["xg"], r["xg"]))
    mains = {e["side"]: e for e in edges if e.get("role") in ("main_L", "main_R")}
    for e in edges:
        if e.get("role") == "split_in" and e["side"] in mains and not mains[e["side"]].get("flare"):  # iter8
            pairs.append((mains[e["side"]]["xg"], e["xg"]))
    # lane vs its side's main edge (exit lane opening between the through-lane line and the widening edge)
    for sd, e in (mains.items() if lane_edge else ()):
        ls = [r for r in lanes if r["side"] == sd and np.isfinite(r["xg"]).sum() >= 8]
        if ls:
            nl = max(ls, key=lambda r: np.nanmedian(r["xg"]) * sd)  # outermost lane line on that side
            pairs.append(("edge", nl["xg"], e["xg"]))
    pairs = [p if isinstance(p[0], str) else ("pair",) + tuple(p) for p in pairs]
    out = []
    for kind, a, b in pairs:
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() < 8:
            continue
        z = ZS[both]; gap = np.abs(b[both] - a[both])
        if kind == "edge":  # the gore is the widening beyond the lane's normal shoulder gap
            zsel = z <= 30.0
            if zsel.sum() < 8:
                continue
            g0 = float(np.percentile(gap[zsel], 10))
            slope = np.polyfit(z[zsel], gap[zsel], 1)[0]
            grow = gap - g0
            sel = zsel & (grow >= 0.3) & (grow <= max_gap)
            if slope < 0.05 or grow[zsel].max() < min_gap or sel.sum() < 4:
                continue
            sgn = np.sign(np.median(b[both] - a[both]))
            xa = a[both][sel] + sgn * g0; xb = b[both][sel]; zz = z[sel]
            poly = np.concatenate([np.stack([xa, zz], 1), np.stack([xb, zz], 1)[::-1]])
            out.append({"poly": poly, "n_stripes": 0, "area": float(_trapz(np.abs(xb - xa), zz)),
                        "centre": poly.mean(0), "src": "fork_edge"})
            continue
        slope = np.polyfit(z, gap, 1)[0]
        if slope <= 0.03 or gap.max() < min_gap:
            continue
        sel = (gap >= 0.3) & (gap <= max_gap)
        if sel.sum() < 4:
            continue
        zz = z[sel]; xa = a[both][sel]; xb = b[both][sel]
        poly = np.concatenate([np.stack([xa, zz], 1), np.stack([xb, zz], 1)[::-1]])
        out.append({"poly": poly, "n_stripes": 0, "area": float(_trapz(np.abs(xb - xa), zz)),
                    "centre": poly.mean(0), "src": "fork"})
    return out


class StageLines:
    """Per-clip lane / edge / gore tracking on BEV masks (road: drivable, paint: markings)."""

    def __init__(self, fps=10.0):
        f = fps  # iter13: fr10 = same time span as at 10 Hz below 10 fps (no-op at >= 10 fps)
        self.lanes = MultiLineTracker("lane", n_birth=fr10(3, f, 2), n_death=fr10(6, f, 2), alpha=0.25)  # iter5: 0.4 -> 0.25
        self.edges = {r: MultiLineTracker("edge", n_birth=fr10(3, f, 2), n_death=fr10(4, f, 2), gate_max=2.0, alpha=0.2)
                      for r in ("main_L", "main_R", "split_in", "split_out")}
        self.gore = GoreTracker(n_birth=fr10(3, f, 2), n_death=fr10(4, f, 2))  # iter6: stripe gores need 3 frames
        self.div = DivergenceGore(n_persist=fr10(3, f, 2), hist=fr10(12, f, 4))
        self.lane_min_hits = fr10(4, f, 2)
        self.n_split_frames = 0
        self.hold = EdgeHold(fps=fps)  # iter8: 0.5 s hold
        self.anchor = LaneAnchor(fps=fps)  # iter12: lane-anchored coast
        self.last_raw = {}; self.last_raw_held = {}
        if not hasattr(self, "kerb"):
            self.kerb = KerbState(fps=fps)

    def step(self, k, road_bev, paint_bev, valid_bev, res, ds=0.0, cars=None):
        self.last_raw = {}; self.last_raw_held = {}
        comps = components(paint_bev, res)
        chains, stripes = extract_chains(comps)
        lanes = clean_lanes([_legacy(r) for r in self.lanes.step(k, [smooth_chain(c) for c in chains])],
                            min_hits=getattr(self, "lane_min_hits", 4))
        road_f = cv2.morphologyEx((road_bev | paint_bev).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        ed = edge_chains(road_f, res, valid=cv2.erode(valid_bev.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        if np.isfinite(ed["split_in"]).any():
            self.n_split_frames += 1
        edges = []
        for role, tr in self.edges.items():
            got = tr.step(k, [ed[role]])
            live = [r for r in got if not r["held"]]
            got = live if live else got[:1]  # one boundary per role: drop a coasting duplicate after a re-birth
            if role.startswith("split"):
                got = [r for r in got if r["hits"] >= 4 and _plausible(r["x"], max_slope=0.25, max_step=0.15)]
            if role.startswith("main"):
                # iter7: trim the bad near-ego / lane-crossing part instead of dropping the edge; hold <= 5 frames
                sd = -1 if role == "main_L" else 1
                if got:  # iter9: keep the untrimmed main edge for the polylines dump (diagnosis of trims on GPU runs)
                    self.last_raw[role] = np.asarray(got[0]["x"], float).copy()
                    self.last_raw_held[role] = bool(got[0].get("held"))  # iter12: tracker coast flag, for exact replays
                tr_ = []
                global TRIM_DEBUG
                if _TRIM_DEBUG_PATH:
                    TRIM_DEBUG = []
                for r in got:
                    xt = trim_edge(r["x"], sd, lanes)
                if _TRIM_DEBUG_PATH and TRIM_DEBUG:
                    with open(_TRIM_DEBUG_PATH, "a") as fh_:
                        for d_ in TRIM_DEBUG:
                            fh_.write(json.dumps({"k": int(k), "role": role, "run": d_["run"], "ref": d_["ref"],
                                                  "x": np.round(d_["x"], 3).tolist(), "bad": d_["bad"].astype(int).tolist(),
                                                  "dev": np.round(d_["dev"], 1).tolist(),
                                                  "lanes": [np.round(l_, 3).tolist() for l_ in d_["lanes"]]}).replace("NaN", "null") + "\n")
                    TRIM_DEBUG = None
                for r in got:
                    xt = trim_edge(r["x"], sd, lanes)
                    if xt is not None:
                        tr_.append(dict(r, x=xt))
                got = tr_
            for r in got:
                side = -1 if role == "main_L" else (1 if role == "main_R" else None)
                edges.append(_legacy(r, side=side, kind="edge", extra={"role": role, "split": role.startswith("split")}))
        edges, fg = self.post_edges(lanes, edges, cars, ds)
        # iter5: stripe gores need >= 3 stripes and a compact hull; plus fork gores (child diverging from parent
        # lane / split edge diverging from main edge with a growing gap)
        sg = [g for g in gore_polygons(stripes, min_stripes=3) if g["area"] <= 40.0]
        gores = [{"poly": np.asarray(g["poly"], float).tolist(), "n_stripes": int(g.get("n_stripes", 0)),
                  "src": g.get("src", "stripes")} for g in self.gore.step(sg + fg)]
        return lanes, edges, gores

    def post_edges(self, lanes, edges, cars=None, ds=0.0):
        """iter11: the part of step() after the edge trackers (kerb -> hold -> retrim -> flare -> divergence gores),
        split out so tools/replay_gore.py replays exactly this code on a GPU run's edges_raw dump."""
        if KERB_CARS:  # iter9: main edge along a parked-car row -> kerb behind the cars (with hysteresis)
            mains = {e.get("role"): e for e in edges if e.get("role") in ("main_L", "main_R")}
            for role, sd in (("main_L", -1), ("main_R", 1)):
                e = mains.get(role)
                if e is None:
                    continue
                o = mains.get("main_R" if role == "main_L" else "main_L")
                xk, app = self.kerb.step(role, e["xg"], sd, cars, _ref_slope(o["xg"] if o is not None else None, lanes), ds, lanes=lanes)
                if app:
                    fk = np.isfinite(xk)
                    c2 = np.polyfit(ZS[fk], xk[fk], 2 if np.ptp(ZS[fk]) > 12 else 1)[::-1]
                    e2 = dict(e, xg=xk, kerb_occ=True, c=np.concatenate([c2, np.zeros(3 - len(c2))]),
                              kerb_alpha=float(self.kerb.alpha.get(role, 1.0)) if self.kerb.ramp_n > 1 else 1.0)
                    edges[edges.index(e)] = e2
        for role, sd in (("main_L", -1), ("main_R", 1)):  # iter7 temporal hold (+ fill of trimmed samples)
            cur = [e for e in edges if e.get("role") == role]
            if cur and is_diag_fragment(cur[0], lanes):  # iter11: a diagonal fragment is no edge -> the hold coasts
                edges.remove(cur[0]); cur = []
                self.n_frag_veto = getattr(self, "n_frag_veto", 0) + 1
            h = self.hold.step(role, cur[0] if cur else None, ds)
            h = retrim_held(h, cur[0] if cur else None, sd, lanes)
            h = frag_extend(h, sd, lanes)  # iter11
            oth = next((e for e in edges if e is not None and e.get("role") == ("main_R" if role == "main_L" else "main_L")), None)
            self.anchor.learn(role, h, sd, lanes, oth)  # iter12: lane-anchored coast
            h0 = h
            h = self.anchor.fill(role, h, sd, lanes, oth)
            h = self._car_clear(h, h0, sd, cars, oth, lanes)  # iter13
            self.anchor.note(role, h)
            if h is not None:
                h["flare"] = is_flared(h.get("xg"), sd, lanes)
            if cur:
                if h is None:  # iter13: a coasted line dropped by the car clearance
                    edges.remove(cur[0])
                else:
                    edges[edges.index(cur[0])] = h
            elif h is not None:
                edges.append(h)
        fg = self.div.step(lanes, edges)
        return edges, fg

    def _car_clear(self, h, h0, side, cars, oth, lanes):
        """iter13: a synthetic main edge (anchored, or an EdgeHold coast) is kept out of car footprints (car_clear).
        A live edge only extended by anchored near samples falls back to the unextended edge if the extension enters a
        car; a fully synthetic line that cannot be pushed clear is dropped (no edge this frame)."""
        if not EDGE_CAR_CLEAR or h is None or not cars:
            return h
        if not _synthetic(h):
            return h
        xc, st = car_clear(h["xg"], side, cars, _ref_slope(oth["xg"] if oth is not None else None, lanes))
        self.n_car = getattr(self, "n_car", {}); self.n_car[st] = self.n_car.get(st, 0) + 1
        if st == "clear":
            return h
        if h.get("anchored") and not h.get("held"):  # live edge + anchored near samples
            return h0 if st == "drop" else _zrange(dict(h, xg=xc, car_push=True), xc)
        if st == "drop":
            return None
        return _zrange(dict(h, xg=xc, car_push=True), xc)

    def events(self):
        return self.lanes.events + [e for t in self.edges.values() for e in t.events]


# ----------------------------------------------------------------------------------------------- iter9
LINE_SMOOTH_S = float(os.environ.get("FBEV_LINE_SMOOTH_S", "0.3"))
LINE_SMOOTH_SYM = os.environ.get("FBEV_LINE_SMOOTH_SYM", "1") == "1"
LINE_SMOOTH_POSE = os.environ.get("FBEV_LINE_SMOOTH_POSE", "0") == "1"  # ego-pose transform (VO yaw) vs travel shift


def _to_frame(x, ego, j, k):
    """line samples x(ZS) in ego frame j -> (x', z') in ego frame k via the ego pose (heading psi, path x/z)."""
    f = np.isfinite(x)
    cj, sj = np.cos(ego["psi"][j]), np.sin(ego["psi"][j])
    X = ego["x"][j] + cj * x[f] + sj * ZS[f]; Zw = ego["z"][j] - sj * x[f] + cj * ZS[f]
    ck, sk = np.cos(ego["psi"][k]), np.sin(ego["psi"][k])
    dx, dz = X - ego["x"][k], Zw - ego["z"][k]
    return ck * dx - sk * dz, sk * dx + ck * dz


def centre_smooth_lines(frames, s, fps, half_s=None, kind="lanes", ego=None):
    """iter9: zero-lag (centred) temporal smoothing of every tracked line, done offline (the whole clip is known
    before rendering). frames[k] = list of line records with 'xg' (ego frame k) and an 'id' (+ 'role' for edges).
    Neighbour frames j within +-half_s seconds are moved into frame k, with the ego pose (VO heading + selected
    travel) when given, else by the travel s only (a static world line at ego z in frame k sits at z - (s[j]-s[k])
    in frame j), and averaged with triangular weights. Only samples present in frame k change: presence, z-extent,
    ids, roles and flags are unchanged. Fallback (legacy) lanes are left alone."""
    half_s = LINE_SMOOTH_S if half_s is None else half_s
    h = int(round(half_s * fps))
    if h < 1:
        return frames
    def key(r):
        if r.get("fallback") or r.get("xg") is None or r.get("id") is None:
            return None
        return (r.get("role"), int(r["id"])) if kind == "edges" else int(r["id"])
    idx = [{key(r): r for r in fr if key(r) is not None} for fr in frames]
    use_pose = ego is not None and all(len(ego[a]) >= len(frames) for a in ("x", "z", "psi"))
    out = []
    for k, fr in enumerate(frames):
        new = []
        for r in fr:
            kk = key(r)
            if kk is None:
                new.append(r); continue
            x0 = np.asarray(r["xg"], float); f0 = np.isfinite(x0)
            if f0.sum() < 2:
                new.append(r); continue
            acc = np.where(f0, x0, 0.0); wsum = f0.astype(float)
            hk = h
            if LINE_SMOOTH_SYM:  # symmetric window only (a one-sided window at a track's birth / end bends a trend)
                hk = 0
                while hk < h and 0 <= k - hk - 1 and k + hk + 1 < len(frames) and kk in idx[k - hk - 1] and kk in idx[k + hk + 1]:
                    hk += 1
            for j in range(max(0, k - hk), min(len(frames), k + hk + 1)):
                rj = idx[j].get(kk) if j != k else None
                if rj is None:
                    continue
                xj = np.asarray(rj["xg"], float); fj = np.isfinite(xj)
                if fj.sum() < 2:
                    continue
                if use_pose:
                    xx, zz = _to_frame(xj, ego, j, k)
                else:
                    xx, zz = xj[fj], ZS[fj] + (float(s[j]) - float(s[k]))  # frame-j z -> frame-k z
                o = np.argsort(zz); xx, zz = xx[o], zz[o]
                if len(zz) < 2 or np.any(np.diff(zz) <= 0):
                    keep = np.concatenate([[True], np.diff(zz) > 1e-6]); xx, zz = xx[keep], zz[keep]
                    if len(zz) < 2:
                        continue
                ok = f0 & (ZS >= zz[0]) & (ZS <= zz[-1])
                if not ok.any():
                    continue
                w = 1.0 - abs(j - k) / (hk + 1.0)
                acc[ok] += w * np.interp(ZS[ok], zz, xx); wsum[ok] += w
            new.append(dict(r, xg=np.where(f0, acc / np.maximum(wsum, 1e-9), np.nan)))
        out.append(new)
    return out



# ----------------------------------------------------------------------------------------------- iter10
PAIR_W_S = float(os.environ.get("FBEV_PAIR_W_S", "1.5"))          # ego lane-pair width window (+- s); 0 = off
PAIR_W_MAXDEV = float(os.environ.get("FBEV_PAIR_W_MAXDEV", "0.5"))  # leave samples whose width moves more alone
PAIR_W_SPLIT = os.environ.get("FBEV_PAIR_W_SPLIT", "resid")         # resid | half: who absorbs the width change
PAIR_W_MODE = os.environ.get("FBEV_PAIR_W_MODE", "kf")          # tri | auto | mean | median; the constant-width priors cut lane_split wjit 0.136 -> 0.06 but cost the final lane change (0.013 -> 0.018)
PAIR_W_TREND_M = float(os.environ.get("FBEV_PAIR_W_TREND_M", "0.3"))  # auto: run width trend above this -> tri
LANE_BACKFILL_S = float(os.environ.get("FBEV_LANE_BACKFILL_S", "0.6"))  # backfill confirmed lanes into the clip start
# iter11: 0.3 -> 0.6 s (GPU iter10 final: the ego-right lane is confirmed at f5, 0.5 s); main road edges too (edge tracker
# n_birth 3: exit / sample / city / lane_split / munich edges start at f2, final R f4, late R f5)
EDGE_BACKFILL_S = float(os.environ.get("FBEV_EDGE_BACKFILL_S", "1.0"))  # iter12: 0.6 -> 1.0 s (a2d2a R edge born at f8)
EDGE_BACKFILL_CV = os.environ.get("FBEV_EDGE_BACKFILL_CV", "1") == "1"
# iter12: offset model of a backfilled edge at z_ref: hold (random walk: the first live offset), cv (iter11: constant
# velocity from the first live frames), travel (pure ego-travel shift). FBEV_EDGE_BACKFILL_CV=0 keeps meaning 'travel'.
EDGE_BACKFILL_COVER = os.environ.get("FBEV_EDGE_BACKFILL_COVER", "1") == "1"  # iter12: source must cover z_ref
EDGE_BACKFILL_MODE = os.environ.get("FBEV_EDGE_BACKFILL_MODE", "hold" if EDGE_BACKFILL_CV else "travel")
# iter11 'kf' mode: the ego pair's width offset at z_ref as a slow random walk (or a local linear trend: width + rate),
# RTS-smoothed per pair run >= PAIR_KF_MIN_S. PAIR_KF_Q: process noise (rw: m/sqrt(s)), PAIR_KF_R: measurement noise (m)
PAIR_KF_Q = float(os.environ.get("FBEV_PAIR_KF_Q", "0.1"))
PAIR_KF_R = float(os.environ.get("FBEV_PAIR_KF_R", "0.15"))
PAIR_KF_MODEL = os.environ.get("FBEV_PAIR_KF_MODEL", "rw")  # trend (local linear trend) | rw (random walk)
PAIR_KF_ZLO = float(os.environ.get("FBEV_PAIR_KF_ZLO", "6.0"))
PAIR_KF_ZHI = float(os.environ.get("FBEV_PAIR_KF_ZHI", "25.0"))
PAIR_KF_MIN_S = float(os.environ.get("FBEV_PAIR_KF_MIN_S", "1.0"))  # shorter pair runs (splits, re-pairing): 'tri'



def kf_trend_smooth(t, y, q=None, r=None, v0=0.5):
    """iter11: RTS smoother for a local-linear-trend series (state [w, w'], w' a random walk with spectral density
    q^2; measurement w + N(0, r^2)); t in seconds, y with nan gaps. Returns the smoothed w (nan where t has no
    support before the first / after the last measurement are still filled - the caller masks)."""
    q = PAIR_KF_Q if q is None else q; r = PAIR_KF_R if r is None else r
    y = np.asarray(y, float); t = np.asarray(t, float); n = len(y)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return y.copy()
    i0 = int(np.flatnonzero(ok)[0])
    xs = np.zeros((n, 2)); Ps = np.zeros((n, 2, 2)); xp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2)); Fs = np.zeros((n, 2, 2))
    rw = PAIR_KF_MODEL == "rw"  # pure random walk: the width itself wanders (q = level noise, m/sqrt(s))
    x = np.array([y[i0], 0.0]); P = np.diag([r * r, 0.0 if rw else v0 * v0])
    H = np.array([1.0, 0.0])
    for k in range(n):
        dt = 0.0 if k == 0 else max(float(t[k] - t[k - 1]), 0.0)
        if rw:
            F = np.eye(2); Q = q * q * np.array([[dt, 0.0], [0.0, 0.0]])
        else:
            F = np.array([[1.0, dt], [0.0, 1.0]])
            Q = q * q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
        if k > 0:
            x = F @ x; P = F @ P @ F.T + Q
        Fs[k] = F; xp[k] = x; Pp[k] = P
        if ok[k]:
            S = float(H @ P @ H) + r * r
            K = (P @ H) / S
            x = x + K * (y[k] - x[0]); P = P - np.outer(K, H @ P)
        xs[k] = x; Ps[k] = P
    for k in range(n - 2, -1, -1):  # Rauch-Tung-Striebel
        Pn = Pp[k + 1]
        C = Ps[k] @ Fs[k + 1].T @ np.linalg.inv(Pn + 1e-12 * np.eye(2))
        xs[k] = xs[k] + C @ (xs[k + 1] - xp[k + 1]); Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pn) @ C.T
    return xs[:, 0]


def _ego_pair(fr, z_ref=10.0):
    """nearest left (x < 0) / right (x >= 0) tracked lane at z_ref (the metrics' definition), as list indices."""
    iz = int(np.argmin(np.abs(ZS - z_ref)))
    L = R = None
    for i, r in enumerate(fr):
        if r.get("fallback") or r.get("xg") is None:
            continue
        v = np.asarray(r["xg"], float)[iz]
        if not np.isfinite(v):
            continue
        if v < 0 and (L is None or v > fr[L]["xg"][iz]):
            L = i
        if v >= 0 and (R is None or v < fr[R]["xg"][iz]):
            R = i
    return L, R


def _tri_smooth(t, v, h):
    """centred triangular smoothing of a series with gaps (None/nan), symmetric window clipped at the ends."""
    v = np.asarray(v, float); out = v.copy(); n = len(v)
    for k in range(n):
        if not np.isfinite(v[k]):
            continue
        hk = min(h, k, n - 1 - k)
        acc = wsum = 0.0
        for j in range(k - hk, k + hk + 1):
            if np.isfinite(v[j]):
                w = 1.0 - abs(j - k) / (hk + 1.0); acc += w * v[j]; wsum += w
        out[k] = acc / wsum
    return out


def pair_width_smooth(frames, s, fps, half_s=None, max_dev=None, split=None, z_ref=10.0, mode=None):
    """iter10 (YOLO gap, lane width jitter: exit 0.023 vs 0.007, late 0.020 vs 0.009, final 0.013 vs 0.008): the
    ego lane pair's spacing wobbles with the dashed-line phase (exit: right line 1.16 <-> 1.98 m, period ~12 frames)
    which +-0.3 s line smoothing cannot remove. Offline joint smoothing of the PAIR, per run of frames with the same
    (left id, right id) pair: the width profile w(z) = x_R - x_L is averaged over +-half_s s (triangular, symmetric,
    clipped to the run; neighbour frames moved by the ego travel as in centre_smooth_lines) - lane width is ~constant
    for one pair - and the change is given to the line that wobbles ('resid': one weight per run, from each line's
    RMS offset from its own +-half_s trend at z_ref; 'half': 50/50). Samples whose width would move > max_dev are
    left alone (a real split / merge). Presence, ids, z-extent unchanged."""
    half_s = PAIR_W_S if half_s is None else half_s
    max_dev = PAIR_W_MAXDEV if max_dev is None else max_dev
    split = PAIR_W_SPLIT if split is None else split
    mode = PAIR_W_MODE if mode is None else mode
    h = int(round(half_s * fps))
    if h < 1 or not frames:
        return frames
    iz = int(np.argmin(np.abs(ZS - z_ref)))
    n = len(frames)
    pairs = [_ego_pair(fr, z_ref) for fr in frames]
    key = [None if (iL is None or iR is None) else (frames[k][iL].get("id"), frames[k][iR].get("id"))
           for k, (iL, iR) in enumerate(pairs)]
    out = [list(fr) for fr in frames]
    k = 0
    while k < n:
        j = k
        while j + 1 < n and key[j + 1] == key[k]:
            j += 1
        if key[k] is not None and j > k:
            idx = list(range(k, j + 1))
            XL = [np.asarray(frames[t][pairs[t][0]]["xg"], float) for t in idx]
            XR = [np.asarray(frames[t][pairs[t][1]]["xg"], float) for t in idx]
            W = [r - l for l, r in zip(XL, XR)]
            if split == "resid":
                vL = np.array([x[iz] for x in XL]); vR = np.array([x[iz] for x in XR])
                eL = np.sqrt(np.nanmean((vL - _tri_smooth(None, vL, h)) ** 2)) if len(idx) > 2 else 0.0
                eR = np.sqrt(np.nanmean((vR - _tri_smooth(None, vR, h)) ** 2)) if len(idx) > 2 else 0.0
                aR = float(eR / (eL + eR)) if eL + eR > 1e-6 else 0.5
            else:
                aR = 0.5
            m = len(idx)
            mode_r = mode
            if mode == "auto":  # constant-width prior only for a run whose width does not trend (a lane change / split does)
                vw = np.array([w_[iz] for w_ in W]); okw = np.isfinite(vw)
                tr_m = abs(np.polyfit(np.flatnonzero(okw), vw[okw], 1)[0]) * (m - 1) if okw.sum() >= 3 else np.inf
                mode_r = "mean" if (tr_m <= PAIR_W_TREND_M and m >= fps) else "tri"  # and >= 1 s of the pair
            if mode_r == "kf" and m < PAIR_KF_MIN_S * fps:
                mode_r = "tri"
            if mode_r == "kf":
                # iter11: the width OFFSET of the profile (LS line over z 6-25 m, evaluated at z_ref) is smoothed as a
                # local linear trend; the profile's shape around that line is kept as measured
                zw = (ZS >= PAIR_KF_ZLO) & (ZS <= PAIR_KF_ZHI)
                a_obs = np.full(m, np.nan); b_obs = np.full(m, np.nan)
                for a_, wk in enumerate(W):
                    fw = np.isfinite(wk) & zw
                    if fw.sum() >= 4:
                        b_obs[a_], a_obs[a_] = np.polyfit(ZS[fw] - z_ref, wk[fw], 1)
                    elif np.isfinite(wk[iz]):
                        a_obs[a_] = wk[iz]; b_obs[a_] = 0.0
                if np.isfinite(a_obs).sum() >= 3:
                    tt = np.array([t / float(fps) for t in idx])
                    a_hat = kf_trend_smooth(tt, a_obs)
                    for a_, t in enumerate(idx):
                        wk = W[a_]; f0 = np.isfinite(wk)
                        if f0.sum() < 2 or not np.isfinite(a_obs[a_]):
                            continue
                        e = np.where(f0, a_hat[a_] - a_obs[a_], 0.0)
                        e = e * np.clip(1.5 - np.abs(e) / max_dev, 0.0, 1.0)
                        iL, iR = pairs[t]
                        out[t][iL] = dict(frames[t][iL], xg=np.where(np.isfinite(XL[a_]), XL[a_] - (1.0 - aR) * e, np.nan))
                        out[t][iR] = dict(frames[t][iR], xg=np.where(np.isfinite(XR[a_]), XR[a_] + aR * e, np.nan))
                k = j + 1
                continue
            for a_, t in enumerate(idx):
                wk = W[a_]; f0 = np.isfinite(wk)
                if f0.sum() < 2:
                    continue
                if mode_r in ("median", "mean"):  # lane-width prior: per-z median of the pair width over the run window
                    lo, hi = max(0, a_ - h), min(m - 1, a_ + h)
                    if hi - lo < 2:
                        continue
                    stack = np.full((hi - lo + 1, len(ZS)), np.nan)
                    for r_, b_ in enumerate(range(lo, hi + 1)):
                        fj = np.isfinite(W[b_])
                        if fj.sum() < 2:
                            continue
                        zz = ZS[fj] + (float(s[idx[b_]]) - float(s[t])); xx = W[b_][fj]
                        ok = f0 & (ZS >= zz[0]) & (ZS <= zz[-1])
                        stack[r_, ok] = np.interp(ZS[ok], zz, xx)
                    cnt = np.isfinite(stack).sum(0)
                    med = np.full(len(ZS), np.nan)
                    okm = f0 & (cnt >= 3)
                    if okm.any():
                        med[okm] = (np.nanmedian if mode_r == "median" else np.nanmean)(stack[:, okm], axis=0)
                    e = np.where(okm, med - wk, 0.0)
                    g = np.clip(1.5 - np.abs(e) / max_dev, 0.0, 1.0)  # full up to max_dev / 2, fades out by 1.5 max_dev
                    e = e * g
                    iL, iR = pairs[t]
                    out[t][iL] = dict(frames[t][iL], xg=np.where(np.isfinite(XL[a_]), XL[a_] - (1.0 - aR) * e, np.nan))
                    out[t][iR] = dict(frames[t][iR], xg=np.where(np.isfinite(XR[a_]), XR[a_] + aR * e, np.nan))
                    continue
                hk = min(h, a_, m - 1 - a_)
                if hk < 1:
                    continue
                acc = np.where(f0, wk, 0.0); wsum = f0.astype(float)
                for b_ in range(a_ - hk, a_ + hk + 1):
                    if b_ == a_:
                        continue
                    fj = np.isfinite(W[b_])
                    if fj.sum() < 2:
                        continue
                    zz = ZS[fj] + (float(s[idx[b_]]) - float(s[t])); xx = W[b_][fj]
                    ok = f0 & (ZS >= zz[0]) & (ZS <= zz[-1])
                    if not ok.any():
                        continue
                    w = 1.0 - abs(b_ - a_) / (hk + 1.0)
                    acc[ok] += w * np.interp(ZS[ok], zz, xx); wsum[ok] += w
                e = np.where(f0, acc / np.maximum(wsum, 1e-9) - wk, 0.0)
                e = e * np.clip(1.5 - np.abs(e) / max_dev, 0.0, 1.0)  # full up to max_dev / 2, fades out by 1.5 max_dev
                iL, iR = pairs[t]
                out[t][iL] = dict(frames[t][iL], xg=np.where(np.isfinite(XL[a_]), XL[a_] - (1.0 - aR) * e, np.nan))
                out[t][iR] = dict(frames[t][iR], xg=np.where(np.isfinite(XR[a_]), XR[a_] + aR * e, np.nan))
        k = j + 1
    return out


def lane_backfill(frames, s, fps, max_s=None, z_ref=10.0, min_hits=4, dup_m=0.8):
    """iter10 (YOLO gap, lane presence: every KITTI clip misses lanes on f0-1 because a lane needs >= 4 hits): offline
    (the clip is processed before rendering), a lane confirmed in the first max_s seconds is copied back into the
    earlier frames of the clip start, moved by the ego travel (a world line at frame-k0 z sits at z + (s[k0] - s[j])
    in frame j) and, when that leaves z_ref uncovered, extended to its frame-k0 near end along its near slope (<= the
    travel, a few metres). Not done when a lane already lies within dup_m at z_ref on that frame. Marked
    'backfill': True. Only the clip start: a mid-clip birth is a real new line."""
    min_hits = fr10(min_hits, fps, 2)  # iter13: time base below 10 fps
    max_s = LANE_BACKFILL_S if max_s is None else max_s
    kmax = int(round(max_s * fps))
    if kmax < 1 or not frames:
        return frames
    out = [list(fr) for fr in frames]
    iz = int(np.argmin(np.abs(ZS - z_ref)))
    first = {}
    for k, fr in enumerate(frames[: kmax + 1]):
        for r in fr:
            if r.get("fallback") or r.get("xg") is None or r.get("id") is None:
                continue
            if int(r["id"]) not in first:
                first[int(r["id"])] = k
    for lid, k0 in first.items():
        if k0 == 0:
            continue
        r0 = next(r for r in frames[k0] if r.get("id") is not None and int(r["id"]) == lid and r.get("xg") is not None)
        x0 = np.asarray(r0["xg"], float); f0 = np.isfinite(x0)
        if f0.sum() < 4 or r0.get("held"):
            continue
        # the lane must be confirmed (still tracked a few frames later), not a one-off
        if sum(1 for fr in frames[k0:k0 + min_hits] for r in fr if r.get("id") is not None and int(r["id"]) == lid) < min(min_hits, len(frames) - k0):
            continue
        zf = ZS[f0]; xf = x0[f0]
        nz = zf <= zf[0] + 4.0
        sl = np.polyfit(zf[nz], xf[nz], 1)[0] if nz.sum() >= 3 else 0.0
        for j in range(0, k0):
            dz = float(s[k0]) - float(s[j])  # >= 0
            zz = zf + dz
            xj = np.full(len(ZS), np.nan)
            m = (ZS >= zz[0]) & (ZS <= zz[-1]); xj[m] = np.interp(ZS[m], zz, xf)
            ext = (ZS >= zf[0] - 1e-6) & (ZS < zz[0])
            xj[ext] = xf[0] + sl * (ZS[ext] - zz[0])
            v = xj[iz]
            if not np.isfinite(v):
                continue
            if any(np.isfinite(np.asarray(r.get("xg") if r.get("xg") is not None else np.full(len(ZS), np.nan), float)[iz])
                   and abs(np.asarray(r["xg"], float)[iz] - v) < dup_m for r in out[j] if r.get("xg") is not None):
                continue
            fz = ZS[np.isfinite(xj)]
            out[j].append(dict(r0, xg=xj, zmin=float(fz.min()), zmax=float(fz.max()), held=False, backfill=True,
                               hits=int(r0.get("hits", 0))))
    return out

def edge_backfill(frames, s, fps, lanes=None, max_s=None, z_ref=10.0, min_hits=4):
    """iter11 (GPU iter10 exit f0000: lanes backfilled, road edges absent): the lane_backfill rule for the MAIN road
    edges, keyed by role (main_L / main_R): an edge confirmed in the first max_s seconds (a live, not held record that
    is present >= min_hits frames from its first frame) is copied back into the earlier frames that have no edge of
    that role, moved by the ego travel and extended to its near end along its near slope; the copy must pass trim_edge
    against that frame's lanes (if given). Marked 'backfill': True."""
    min_hits = fr10(min_hits, fps, 2)  # iter13: time base below 10 fps
    max_s = EDGE_BACKFILL_S if max_s is None else max_s
    kmax = int(round(max_s * fps))
    if kmax < 1 or not frames:
        return frames
    out = [list(fr) for fr in frames]
    iz = int(np.argmin(np.abs(ZS - z_ref)))
    for role, sd in (("main_L", -1), ("main_R", 1)):
        k0 = r0 = None; seen = False
        for k, fr in enumerate(frames[: kmax + 1]):
            r = next((e for e in fr if e.get("role") == role and e.get("xg") is not None), None)
            if r is not None:
                if r.get("held"):
                    if EDGE_BACKFILL_COVER and seen:
                        continue  # a coasted frame between partial live ones (munich main_R f13)
                    break
                seen = True
                # iter12 (munich main_R starts at z 17 m until f8 and reaches z 10 m at f15): the source is the first
                # live edge that covers z_ref; earlier records of the role only get their missing near samples
                if EDGE_BACKFILL_COVER and not np.isfinite(np.asarray(r["xg"], float)[iz]):
                    continue
                k0, r0 = k, r
                break
        if not k0 or r0 is None:
            continue
        if sum(1 for fr in frames[k0:k0 + min_hits] if any(e.get("role") == role for e in fr)) < min(min_hits, len(frames) - k0):
            continue
        x0 = np.asarray(r0["xg"], float); f0 = np.isfinite(x0)
        if f0.sum() < 4:
            continue
        zf = ZS[f0]; xf = x0[f0]
        nz = zf <= zf[0] + 4.0
        sl = np.polyfit(zf[nz], xf[nz], 1)[0] if nz.sum() >= 3 else 0.0
        # the edge's own lateral motion at z_ref over its first live frames (constant-velocity continuation: the pure
        # travel shift over-extrapolates a diverging edge pair, city 0.31 vs 0.18 m / frame width steps)
        iz = int(np.argmin(np.abs(ZS - z_ref)))
        vk, vx = [], []
        for k in range(k0, min(len(frames), k0 + min_hits)):
            r = next((e for e in frames[k] if e.get("role") == role and e.get("xg") is not None and not e.get("held")), None)
            if r is not None and np.isfinite(np.asarray(r["xg"], float)[iz]):
                vk.append(k); vx.append(float(np.asarray(r["xg"], float)[iz]))
        vel = float(np.polyfit(vk, vx, 1)[0]) if len(vk) >= 3 else None
        for j in range(0, k0):
            ej = next((e for e in out[j] if e.get("role") == role), None)
            if ej is not None and (not EDGE_BACKFILL_COVER or ej.get("xg") is None
                                   or np.isfinite(np.asarray(ej["xg"], float)[iz])):
                continue
            dz = float(s[k0]) - float(s[j])
            zz = zf + dz
            xj = np.full(len(ZS), np.nan)
            m = (ZS >= zz[0]) & (ZS <= zz[-1]); xj[m] = np.interp(ZS[m], zz, xf)
            ext = (ZS >= zf[0] - 1e-6) & (ZS < zz[0])
            xj[ext] = xf[0] + sl * (ZS[ext] - zz[0])
            if EDGE_BACKFILL_MODE == "hold" and np.isfinite(xj[iz]) and np.isfinite(x0[iz]):
                # iter12: random-walk prior (as the iter11 pair width): before its first measurement the edge's best
                # offset at z_ref is that measurement; the CV continuation ran city's L edge 0.4 m further onto the
                # parked cars / across the lane line and raised the edge wjit (city 0.054 -> 0.076, late 0.035 -> 0.041)
                xj = xj + float(np.clip(x0[iz] - xj[iz], -0.5, 0.5))
            elif EDGE_BACKFILL_MODE == "cv" and vel is not None and np.isfinite(xj[iz]) and np.isfinite(x0[iz]):
                xj = xj + float(np.clip(x0[iz] - vel * (k0 - j) - xj[iz], -0.5, 0.5))
            if lanes is not None and j < len(lanes):
                lj = [r for r in lanes[j] if r.get("xg") is not None and not r.get("fallback")]
                xt = trim_edge(xj, sd, lj)
                if xt is None:
                    continue
                xj = xt
            fj = np.isfinite(xj)
            if fj.sum() < 4:
                continue
            if EDGE_BACKFILL_COVER:  # iter12: no diagonal / kinked copy (munich f0: a 14-frame shift of a diagonal edge)
                lj = [r for r in lanes[j] if r.get("xg") is not None and not r.get("fallback")] if lanes is not None and j < len(lanes) else []
                nzm = fj & (ZS <= 20.0)
                if nzm.sum() >= 4 and abs(float(np.polyfit(ZS[nzm], xj[nzm], 1)[0]) - _ref_slope(None, lj)) > EDGE_FRAG_DEV:
                    continue
            if ej is not None:  # iter12: near fill of a partial edge: add only the samples below its near end
                ex = np.asarray(ej["xg"], float); fe = np.isfinite(ex)
                if fe.sum() < 2:
                    continue
                i0 = int(np.flatnonzero(fe)[0])
                if not np.isfinite(xj[i0]) or abs(xj[i0] - ex[i0]) > 0.6:
                    continue
                xn = ex.copy(); m = (np.arange(len(ZS)) < i0) & np.isfinite(xj)
                xn[m] = xj[m] + (ex[i0] - xj[i0])
                if lanes is not None and j < len(lanes):
                    xt = trim_edge(xn, sd, [r for r in lanes[j] if r.get("xg") is not None and not r.get("fallback")])
                    if xt is None or (fe & ~np.isfinite(xt)).any():
                        continue
                    xn = xt
                fz = ZS[np.isfinite(xn)]
                out[j][out[j].index(ej)] = dict(ej, xg=xn, zmin=float(fz.min()), zmax=float(fz.max()), backfill=True)
                continue
            fz = ZS[fj]
            out[j].append(dict(r0, xg=xj, zmin=float(fz.min()), zmax=float(fz.max()), held=False, backfill=True,
                               kerb_occ=False, kerb_alpha=0.0))
    return out


# iter13: clip-start edges anchored to the frame's own lane / opposite edge (frame-0 completeness) when the iter11-12
# copy of the first live edge is rejected (diagonal / kinked) or leaves a partial edge short of z_ref
EDGE_ANCHOR_BACKFILL = os.environ.get("FBEV_EDGE_ANCHOR_BACKFILL", "1") == "1"


def _anchor_offset(e, side, lanes, other, iz):
    """offset of a live main edge at ZS[iz] against (a) the nearest same-side lane (>= 0.5 m outside it), else (b) the
    opposite main edge (>= 2.5 m), each only if the two run parallel (|slope diff| <= EDGE_ANCHOR_DEV over z <= 25 m).
    Returns (kind, offset) or None. (LaneAnchor's learning rule, for the offline clip-start pass.)"""
    x = np.asarray(e["xg"], float)
    if not np.isfinite(x[iz]):
        return None
    for kind, ref, mn in (("lane", _nearest_lane_x(lanes, side, ZS[iz]), 0.5),
                          ("edge", None if other is None else np.asarray(other["xg"], float), 2.5)):
        if ref is None:
            continue
        m = np.isfinite(x) & np.isfinite(ref) & (ZS <= 25.0)
        if m.sum() < 8 or not np.isfinite(ref[iz]):
            continue
        if abs(float(np.polyfit(ZS[m], x[m] - ref[m], 1)[0])) > EDGE_ANCHOR_DEV:
            continue
        off = float(x[iz] - ref[iz])
        if side * off >= mn:
            return kind, off
    return None


def edge_anchor_backfill(frames, s, fps, lanes, max_s=None, z_ref=10.0):
    """iter13 (frame 0 on munich R, a2d2b L): for each main role, the frames before its first live edge reaching z_ref
    (within max_s = EDGE_BACKFILL_S) that still have no edge at z_ref get reference + offset: the offset of that first
    live edge against its nearest same-side lane (else the opposite main edge), if parallel (_anchor_offset); the
    reference is the same lane kind in frame j (nearest same-side lane / the opposite main edge there). The line must
    pass trim_edge and reach z_ref. A partial edge in frame j only gets the samples below its near end, shifted to join
    it, if the gap there is <= 0.6 m. Marked backfill + anchored."""
    max_s = EDGE_BACKFILL_S if max_s is None else max_s
    kmax = int(round(max_s * fps))
    if not EDGE_ANCHOR_BACKFILL or kmax < 1 or not frames or lanes is None:
        return frames
    out = [list(fr) for fr in frames]
    iz = int(np.argmin(np.abs(ZS - z_ref)))

    def rec(fr, role):
        return next((e for e in fr if e.get("role") == role and e.get("xg") is not None), None)

    def lane_list(j):
        return [r for r in lanes[j] if r.get("xg") is not None and not r.get("fallback")] if j < len(lanes) else []

    for role, sd in (("main_L", -1), ("main_R", 1)):
        orole = "main_R" if role == "main_L" else "main_L"
        k1 = None
        for k, fr in enumerate(out[: kmax + 1]):
            r = rec(fr, role)
            if r is not None and np.isfinite(np.asarray(r["xg"], float)[iz]):
                k1 = k
                break
        if not k1:
            continue  # present at frame 0 already, or never within the window
        src = None
        for k in range(k1, min(len(out), k1 + int(round(0.5 * fps)) + 1)):  # first parallel live edge within 0.5 s
            r = rec(frames[k], role)
            if r is None or r.get("held") or r.get("backfill") or r.get("anchored") or r.get("kerb_occ"):
                continue
            o = rec(frames[k], orole)
            got = _anchor_offset(r, sd, lane_list(k), o if o is not None and not o.get("held") else None, iz)
            if got is not None:
                src = (k, r, got)
                break
        if src is None:
            continue
        _, r0, (kind, off) = src
        for j in range(k1):
            ej = rec(out[j], role)
            if ej is not None and np.isfinite(np.asarray(ej["xg"], float)[iz]):
                continue
            lj = lane_list(j)
            if kind == "lane":
                ref = _nearest_lane_x(lj, sd, z_ref)
            else:
                o = rec(out[j], orole); ref = None if o is None else np.asarray(o["xg"], float)
            if ref is None or not np.isfinite(ref[iz]):
                continue
            x = np.where(np.isfinite(ref), ref + off, np.nan); x[ZS > 40.0] = np.nan
            if ej is not None:
                ex = np.asarray(ej["xg"], float); fe = np.isfinite(ex)
                if fe.sum() < 2:
                    continue
                i0 = int(np.flatnonzero(fe)[0])
                if not np.isfinite(x[i0]) or abs(x[i0] - ex[i0]) > 0.6:
                    continue
                xn = ex.copy(); m = (np.arange(len(ZS)) < i0) & np.isfinite(x) & (ZS >= z_ref - 1e-6)
                xn[m] = x[m] + (ex[i0] - x[i0])
                xt = trim_edge(xn, sd, lj)
                if xt is None or not np.isfinite(xt[iz]) or (fe & ~np.isfinite(xt)).any():
                    continue
                fz = ZS[np.isfinite(xt)]
                out[j][out[j].index(ej)] = dict(ej, xg=xt, zmin=float(fz.min()), zmax=float(fz.max()), backfill=True,
                                                anchored=True)
                continue
            xt = trim_edge(x, sd, lj)
            if xt is None or not np.isfinite(xt[iz]) or np.isfinite(xt).sum() < 4:
                continue
            fz = ZS[np.isfinite(xt)]
            out[j].append(dict(r0, xg=xt, zmin=float(fz.min()), zmax=float(fz.max()), held=False, backfill=True,
                               anchored=True, kerb_occ=False, kerb_alpha=0.0, c=np.zeros(3)))
    return out


def edge_car_clear(frames, cars, lanes=None):
    """iter13: the final pass (after centre smoothing and the clip-start backfills, which can move a cleared line back
    into a car: munich R f174-177, f242) of car_clear over every SYNTHETIC main edge: anchored, EdgeHold-coasted
    (hold_age) or backfilled. cars[k] = car_footprints of frame k. Pushed lines get car_push=True; lines that cannot be
    pushed clear are removed. Live and kerb-pushed measurements are not touched."""
    if not EDGE_CAR_CLEAR or cars is None:
        return frames
    out = []
    for k, fr in enumerate(frames):
        ck = cars[k] if k < len(cars) else None
        if not ck:
            out.append(fr); continue
        lk = [r for r in (lanes[k] if lanes is not None and k < len(lanes) else []) if r.get("xg") is not None and not r.get("fallback")]
        row = []
        for e in fr:
            role = e.get("role")
            if role not in ("main_L", "main_R") or not _synthetic(e) or e.get("xg") is None:
                row.append(e); continue
            sd = -1 if role == "main_L" else 1
            o = next((x for x in fr if x.get("role") == ("main_R" if sd < 0 else "main_L") and x.get("xg") is not None), None)
            xc, st = car_clear(e["xg"], sd, ck, _ref_slope(o["xg"] if o is not None else None, lk))
            if st == "drop":
                continue
            row.append(e if st == "clear" else _zrange(dict(e, xg=xc, car_push=True), xc))
        out.append(row)
    return out


KERB_CARS = os.environ.get("FBEV_KERB_CARS", "1") == "1"
KERB_MAX_PUSH_M = float(os.environ.get("FBEV_KERB_MAX_PUSH_M", "2.5"))
KERB_MARGIN_M = float(os.environ.get("FBEV_KERB_MARGIN_M", "0.0"))
KERB_CAR_LEN_M = 4.5
# iter12: traffic guard. 'outer' (iter9-11): a long lane line >= 1 m inside the car's outer flank = traffic. 'under':
# unless a long line also runs UNDER the car (between its flanks, tol inside each): city's cars park over the
# parking-lane line, so that row is parked, not a queue.
KERB_TRAFFIC = os.environ.get("FBEV_KERB_TRAFFIC", "outer")  # under: tried, pushes city L to -9.8 m (box flanks too wide)
KERB_TRAFFIC_TOL = float(os.environ.get("FBEV_KERB_TRAFFIC_TOL", "0.3"))


def car_footprints(boxes, lift, inner_frac=0.15):
    """iter9: image car boxes -> BEV contact rows. lift(us, vs) -> (N, 2) ground x, z. For each box: the ground row of
    its bottom edge zb, the flank x at that row (xl, xr) and the central part (xa, xb: 15 % in from each side) that
    marks 'the edge runs along this car's bottom'."""
    out = []
    for b in boxes or []:
        x1, _, x2, y2 = [float(v) for v in b[:4]]
        wb = x2 - x1
        p = lift([x1, x2, x1 + inner_frac * wb, x2 - inner_frac * wb], [y2] * 4)
        if not np.isfinite(p).all() or p[0, 1] <= 1.0:
            continue
        out.append({"zb": float(np.median(p[:, 1])), "xl": float(p[0, 0]), "xr": float(p[1, 0]),
                    "xa": float(p[2, 0]), "xb": float(p[3, 0])})
    return out


def _ref_slope(other_x, lanes):
    """road heading as dx/dz: the opposite main edge near the ego (z <= 25 m), else the median lane heading, else 0."""
    if other_x is not None:
        o = np.asarray(other_x, float); f = np.isfinite(o) & (ZS <= 25.0)
        if f.sum() >= 8:
            return float(np.polyfit(ZS[f], o[f], 1)[0])
    la = [v for v in (_near_angle(xg_of(r)) for r in lanes) if v is not None]
    return float(np.tan(np.radians(np.median(la)))) if la else 0.0


def kerb_behind_cars(x, side, cars, ref_slope, max_push=None, margin=None, car_len=KERB_CAR_LEN_M, line_only=False,
                     lanes=None):
    """iter9 (lane_split f29, munich R): next to a row of parked cars the SAM drivable-area boundary is the cars'
    bottoms / shadows, so the main edge runs diagonally in front of them instead of along the kerb behind them.
    If the edge runs along the bottom of >= 1 car on its side (>= 2 samples within [zb - 0.5, zb + 1.5] whose
    bearing falls inside the central 70 % of that car's bottom), the kerb is taken as the straight line with the
    road heading (ref_slope) that clears the outer flank of every car on that side (+ margin), and the edge becomes
    the outer envelope max(edge, line) up to the last such car + car_len + 3 m (the push ramps out over 3 m after).
    Cars whose outer flank is > max_push outside the edge are ignored (beyond the kerb: driveway, other road).
    Only existing samples move, and only outward, so presence and z-extent are unchanged and the edge cannot cross
    a lane or enter the ego corridor through this. Returns (x, applied)."""
    max_push = KERB_MAX_PUSH_M if max_push is None else max_push
    margin = KERB_MARGIN_M if margin is None else margin
    if x is None or not cars:
        return x, False
    x = np.asarray(x, float); f = np.isfinite(x)
    if f.sum() < 4:
        return x, False
    q = side * x  # outward-positive
    zmin, zmax = ZS[f].min(), ZS[f].max()
    bounded, cons = False, []
    lx = [np.asarray(xg_of(r), float) for r in (lanes or []) if not r.get("fallback") and xg_of(r) is not None]
    for c in cars:
        outer = c["xr"] if side > 0 else c["xl"]
        if side * outer < 1.0 or not (zmin - 0.5 <= c["zb"] <= zmax + 0.5):
            continue
        # traffic, not parking (munich f692 / f760 queues): a long (>= 8 m) painted lane line runs alongside the car,
        # >= 1 m inside its outer flank (a parked row has no lane line under it; lane_split's bay line is short)
        wz = (ZS >= c["zb"] - 1.0) & (ZS <= c["zb"] + car_len)
        traffic = False
        inner = c["xl"] if side > 0 else c["xr"]
        under = False
        for l in lx:
            fl = np.isfinite(l)
            if fl.sum() < 2 or np.ptp(ZS[fl]) < 8.0 or not (fl & wz).any():
                continue
            v = float(np.median(side * l[fl & wz]))
            if 0.5 < v < side * outer - 1.0:
                traffic = True
            if side * inner + KERB_TRAFFIC_TOL < v < side * outer - KERB_TRAFFIC_TOL:
                under = True
        if traffic and under and KERB_TRAFFIC == "under":
            traffic = False  # iter12: the car sits over a long painted line (parked over the bay line): not traffic
        if traffic:
            continue
        m = f & (ZS >= c["zb"] - 0.5) & (ZS <= c["zb"] + 1.5)
        if m.any():
            ra, rb = sorted((c["xa"] / c["zb"], c["xb"] / c["zb"]))
            r = x[m] / ZS[m]
            if int(((r >= ra) & (r <= rb)).sum()) >= 2:
                bounded = True
        qz = float(np.interp(c["zb"], ZS[f], q[f]))
        qo = side * outer + margin
        if qo > qz and qo - qz <= max_push:
            cons.append((qo, c["zb"]))
    if not bounded or not cons:
        return x, False
    B = side * ref_slope
    A = max(qo - B * zb for qo, zb in cons)
    z_end = max(zb for _, zb in cons) + car_len + 3.0
    if line_only:
        return (A, B, z_end), True
    return kerb_apply(x, side, A, B, z_end, max_push)


def kerb_apply(x, side, A, B, z_end, max_push=None):
    """push the edge outward onto the kerb line q = A + B z (outward-positive), capped at max_push, ramped out over
    3 m after z_end. Only finite samples move, only outward. Returns (x, applied)."""
    max_push = KERB_MAX_PUSH_M if max_push is None else max_push
    x = np.asarray(x, float); f = np.isfinite(x)
    if f.sum() < 2:
        return x, False
    q = side * x
    push = np.clip(A + B * ZS - q, 0.0, max_push) * np.clip(1.0 - (ZS - z_end) / 3.0, 0.0, 1.0)
    qn = np.where(f, q + push, np.nan)
    return side * qn, bool((push[f] > 0.05).any())


EDGE_CAR_CLEAR = os.environ.get("FBEV_EDGE_CAR_CLEAR", "1") == "1"  # iter13: synthetic edges never inside a car
EDGE_CAR_TOL = float(os.environ.get("FBEV_EDGE_CAR_TOL", "0.2"))
# the car's lateral extent = its inner (ego-side) box flank + EDGE_CAR_W: for an obliquely seen parked car the box also
# spans the visible side, so the far flank lifted at the contact row lies metres beyond the car (city: -9.3 m for a car
# whose inner flank is at -4.9 m)
EDGE_CAR_W = float(os.environ.get("FBEV_EDGE_CAR_W", "2.0"))
EDGE_CAR_COAST = os.environ.get("FBEV_EDGE_CAR_COAST", "0") == "1"  # iter13: 1 = also car-clear tracker-coasted edges (rejected: A2D2 edge wjit +0.03)


def _synthetic(e):
    """no measurement this frame: anchored, backfilled, EdgeHold-coasted (hold_age) or BEV-tracker-coasted (held)."""
    if e.get("anchored") or e.get("backfill"):
        return True
    return bool(e.get("held")) and ("hold_age" in e or EDGE_CAR_COAST)


def _car_q(c, side):
    """outward-positive (inner, outer) of a car footprint: inner box flank, outer = min(box far flank, inner + W)."""
    inner, outer = (c["xl"], c["xr"]) if side > 0 else (c["xr"], c["xl"])
    return side * inner, min(side * outer, side * inner + EDGE_CAR_W)


def car_inside(x, side, cars, tol=None, car_len=KERB_CAR_LEN_M):
    """iter13: the cars (on the edge's side) whose BEV footprint the line enters: a sample at z in [zb - 0.5,
    zb + car_len] lies strictly inside the car's lateral extent (_car_q: inner flank .. inner + EDGE_CAR_W), tol m
    from each side (outward-positive). Returns the list of such cars."""
    tol = EDGE_CAR_TOL if tol is None else tol
    if x is None or not cars:
        return []
    x = np.asarray(x, float); f = np.isfinite(x); q = side * x
    hit = []
    for c in cars:
        qi, qo = _car_q(c, side)
        if qo < 1.0:
            continue
        lo, hi = qi + tol, qo - tol
        m = f & (ZS >= c["zb"] - 0.5) & (ZS <= c["zb"] + car_len)
        if m.any() and ((q[m] > lo) & (q[m] < hi)).any():
            hit.append(c)
    return hit


EDGE_CAR_PUSH = os.environ.get("FBEV_EDGE_CAR_PUSH", "line")  # iter13: 'line' (kerb geometry first) or 'local' first


def _push_line(x, side, cars, ref_slope, max_push, car_len):
    """the kerb rule's geometry: the road-heading line clearing the outer sides of the cars the line enters (re-checked
    against all cars up to 3 times: a push out of one car can enter the next one)."""
    B = side * ref_slope
    hit = car_inside(x, side, cars); used = []
    for _ in range(3):
        used += [c for c in hit if c not in used]
        # the line must clear each car over its whole z range: bind at the near end (zb - 0.5) for an outward road
        # heading, at the far end (zb + car_len) for an inward one (city L curves back in at 30-40 m)
        A = max(_car_q(c, side)[1] + EDGE_CAR_TOL - B * (c["zb"] + car_len if B < 0 else c["zb"] - 0.5) for c in used)
        z_end = max(c["zb"] for c in used) + car_len + 3.0
        xp, _ = kerb_apply(x, side, A, B, z_end, max_push)  # always from the original line (one capped push)
        hit = car_inside(xp, side, cars)
        if not hit:
            return xp
    return None


def _push_local(x, side, cars, max_push, car_len):
    """only over the z range of each car the line enters (+ 3 m linear ramps): the line can pass between two rows
    (lane_split f0: a parked car in front of a second one further out)."""
    x = np.asarray(x, float); f = np.isfinite(x); q = side * x
    mp = KERB_MAX_PUSH_M if max_push is None else max_push
    tgt = np.full(len(ZS), -np.inf)
    for c in car_inside(x, side, cars):
        qo = _car_q(c, side)[1] + EDGE_CAR_TOL
        z0, z1 = c["zb"] - 0.5, c["zb"] + car_len
        w = np.clip(1.0 - np.maximum(z0 - ZS, ZS - z1).clip(0.0) / 3.0, 0.0, 1.0)
        tgt = np.maximum(tgt, np.where(w > 0, q + w * (qo - q), -np.inf))
    push = np.clip(np.where(np.isfinite(tgt), tgt - q, 0.0), 0.0, mp)
    xl = np.where(f, side * (q + push), np.nan)
    return None if car_inside(xl, side, cars) else xl


def car_clear(x, side, cars, ref_slope, max_push=None, car_len=KERB_CAR_LEN_M):
    """iter13 (GPU iter12 munich R f190-196, f250-251 anchored line, f13 backfill, a2d2b L f14): a SYNTHETIC main edge
    (anchored, EdgeHold-coasted, backfilled: no measurement this frame) must not run through a parked car. If it
    enters cars' footprints (car_inside), it is pushed out: the kerb rule's road-heading line clearing those cars'
    outer sides (_push_line), else a local push over each car's z range (_push_local); EDGE_CAR_PUSH='local' tries
    them the other way round. Only outward, capped at max_push.
    Returns (x, status): 'clear' (untouched), 'pushed', or 'drop' (no push clears it: no edge)."""
    if not car_inside(x, side, cars):
        return x, "clear"
    order = (_push_line, _push_local) if EDGE_CAR_PUSH == "line" else (_push_local, _push_line)
    for fn in order:
        xp = fn(x, side, cars, ref_slope, max_push, car_len) if fn is _push_line else fn(x, side, cars, max_push, car_len)
        if xp is not None:
            return xp, "pushed"
    if EDGE_CAR_TRIM:
        xt = _trim_before_cars(x, side, cars, car_len)
        if xt is not None:
            return xt, "trimmed"
    return x, "drop"


EDGE_CAR_TRIM = os.environ.get("FBEV_EDGE_CAR_TRIM", "1") == "1"  # iter13: cut short in front of the car instead of dropping
EDGE_CAR_TRIM_MIN = float(os.environ.get("FBEV_EDGE_CAR_TRIM_MIN", "4.0"))  # m of line that must remain


def _trim_before_cars(x, side, cars, car_len):
    """iter13 (CPU smoke lane_split R f1-2: a backfilled line that no push clears was dropped, -6.7 pts presence):
    end the synthetic line 0.5 m before the nearest car it enters (repeated while it still enters one). Kept only if
    >= EDGE_CAR_TRIM_MIN m remain, else None."""
    xt = np.array(x, float)
    for _ in range(4):
        hit = car_inside(xt, side, cars, car_len=car_len)
        if not hit:
            f = np.isfinite(xt)
            return xt if f.any() and ZS[f].max() - ZS[f].min() >= EDGE_CAR_TRIM_MIN else None
        xt[ZS >= min(c["zb"] for c in hit) - 0.5] = np.nan
    return None


def _zrange(e, xc):
    f = np.isfinite(np.asarray(xc, float))
    return dict(e, zmin=float(ZS[f].min()), zmax=float(ZS[f].max())) if f.any() else e


KERB_ENTER = int(os.environ.get("FBEV_KERB_ENTER", "3"))  # consecutive bounded frames before the kerb line applies
KERB_HOLD = int(os.environ.get("FBEV_KERB_HOLD", "5"))    # frames the kerb line is carried after the last bounded one
KERB_MED_N = 3                                            # kerb line = median of the last N bounded lines
KERB_RAMP_S = float(os.environ.get("FBEV_KERB_RAMP_S", "0.3"))  # iter10: blend in / out time (0 = iter9 switch)
KERB_HOLD_HITS = 5                                        # a hold only after >= this many bounded frames


class KerbState:
    """iter9: temporal hysteresis for kerb_behind_cars (the per-frame rule toggled with car-box flicker and made the
    edge jump by metres on a2d2a / munich): enter after KERB_ENTER consecutive bounded frames (short spurious bursts on
    munich traffic queues stay off); the line used is the
    median (offset at z = 10 m, slope, end) of the last KERB_MED_N bounded lines, each carried with the ego travel ds;
    after the last bounded frame the line is carried for KERB_HOLD frames (only if >= KERB_HOLD_HITS bounded frames),
    then dropped."""

    ZR = 10.0

    def __init__(self, fps=10.0):
        self.s = {}
        # iter10: blend the kerb push in / out over KERB_RAMP_S (the on/off switch made edge wjit regress:
        # lane_split 0.107 -> 0.130, a2d2a 0.062 -> 0.078)
        self.ramp_n = max(1, int(round(KERB_RAMP_S * fps)))
        self.alpha = {}
        self.last_line = {}

    def step(self, role, x, side, cars, ref_slope, ds=0.0, lanes=None):
        xo, app = self._step(role, x, side, cars, ref_slope, ds, lanes)
        a0 = self.alpha.get(role, 0.0)
        if self.ramp_n <= 1:
            return xo, app
        if app:
            a1 = min(1.0, a0 + 1.0 / self.ramp_n)
            self.last_line[role] = self._cur_line(role)
        else:
            a1 = max(0.0, a0 - 1.0 / self.ramp_n)
        self.alpha[role] = a1
        if x is None or a1 <= 0.0:
            self.last_line.pop(role, None) if a1 <= 0.0 else None
            return xo, False
        if app:
            xa = np.asarray(x, float)
            return np.where(np.isfinite(xa), xa + a1 * (np.asarray(xo, float) - xa), np.nan), True
        ln = self.last_line.get(role)  # ramp-out: the last kerb line, carried with the ego travel
        if ln is None:
            return xo, False
        q, B, ze = ln
        q = q + B * ds; ze = ze - ds
        self.last_line[role] = (q, B, ze)
        xk, ok = kerb_apply(x, side, q - B * self.ZR, B, ze)
        if not ok:
            return xo, False
        xa = np.asarray(x, float)
        return np.where(np.isfinite(xa), xa + a1 * (np.asarray(xk, float) - xa), np.nan), True

    def _cur_line(self, role):
        st = self.s.get(role)
        if st is None or not st["L"]:
            return None
        return tuple(float(np.median([l[i] for l in st["L"]])) for i in range(3))

    def _step(self, role, x, side, cars, ref_slope, ds=0.0, lanes=None):
        st = self.s.get(role)
        if st is not None:  # ego moved forward ds: a world line at frame-k z sits at z - ds now
            st["L"] = [(q + B * ds, B, ze - ds) for q, B, ze in st["L"]]
        ln, ok = (kerb_behind_cars(x, side, cars, ref_slope, line_only=True, lanes=lanes) if x is not None
                  else (None, False))
        if ok:
            A, B, z_end = ln
            if st is None:
                st = {"L": [], "hits": 0, "age": 0}
            st["L"] = (st["L"] + [(A + B * self.ZR, B, z_end)])[-KERB_MED_N:]
            st["hits"] += 1; st["age"] = 0
        elif st is not None:
            st["age"] += 1
            if st["age"] > KERB_HOLD or st["hits"] < KERB_HOLD_HITS:
                st = None
        if st is None:
            self.s.pop(role, None)
            return x, False
        self.s[role] = st
        if st["hits"] < KERB_ENTER or x is None:
            return x, False
        q, B, ze = (float(np.median([l[i] for l in st["L"]])) for i in range(3))
        return kerb_apply(x, side, q - B * self.ZR, B, ze)
