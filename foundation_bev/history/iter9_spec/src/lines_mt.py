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
import os

import numpy as np

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
    la = [v for v in (_near_angle(xg_of(r)) for r in lanes) if v is not None]
    ref = np.radians(float(np.median(la))) if la else 0.0
    xs = np.where(f, x, np.interp(ZS, ZS[f], x[f]))
    dev = np.degrees(np.arctan(np.gradient(xs, ZS)) - ref) * side  # > 0: diverging (moving outward with distance)
    near = f & (ZS <= z_near)
    bad |= near & (((dev < 0) & (-dev > max_deg)) | ((dev > 0) & (dev > div_deg)))
    run = _nearest_run(f & ~bad, int(round(min_len_m / (ZS[1] - ZS[0]))))
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
        self.max_age = max_age if max_age is not None else max(5, int(round(hold_s * fps)))
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


class DivergenceGore:
    """iter6: persistent divergence gores. A candidate (per-frame fork_gores wedge, or a TEMPORAL widening of the
    gap between the outermost lane line and that side's main edge) must persist >= n_persist consecutive frames
    with a non-decreasing gap before it is drawn."""

    def __init__(self, n_persist=3, grow_m=float(os.environ.get("FBEV_GORE_GROW_M", 1.2)), hist=12,
                 max_step=float(os.environ.get("FBEV_GORE_MAX_STEP", 0.4))):
        self.n_persist, self.grow_m, self.hist, self.max_step = n_persist, grow_m, hist, max_step
        self.run = {}      # key -> consecutive frames
        self.gaps = {-1: [], 1: []}

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
        self.lanes = MultiLineTracker("lane", alpha=0.25)  # iter5: 0.4 -> 0.25
        self.edges = {r: MultiLineTracker("edge", n_birth=3, n_death=4, gate_max=2.0, alpha=0.2)
                      for r in ("main_L", "main_R", "split_in", "split_out")}
        self.gore = GoreTracker(n_birth=3)  # iter6: stripe gores need 3 frames
        self.div = DivergenceGore()
        self.n_split_frames = 0
        self.hold = EdgeHold(fps=fps)  # iter8: 0.5 s hold
        self.last_raw = {}
        if not hasattr(self, "kerb"):
            self.kerb = KerbState()

    def step(self, k, road_bev, paint_bev, valid_bev, res, ds=0.0, cars=None):
        self.last_raw = {}
        comps = components(paint_bev, res)
        chains, stripes = extract_chains(comps)
        lanes = clean_lanes([_legacy(r) for r in self.lanes.step(k, [smooth_chain(c) for c in chains])])
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
                tr_ = []
                for r in got:
                    xt = trim_edge(r["x"], sd, lanes)
                    if xt is not None:
                        tr_.append(dict(r, x=xt))
                got = tr_
            for r in got:
                side = -1 if role == "main_L" else (1 if role == "main_R" else None)
                edges.append(_legacy(r, side=side, kind="edge", extra={"role": role, "split": role.startswith("split")}))
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
                    e2 = dict(e, xg=xk, kerb_occ=True, c=np.concatenate([c2, np.zeros(3 - len(c2))]))
                    edges[edges.index(e)] = e2
        for role, sd in (("main_L", -1), ("main_R", 1)):  # iter7 temporal hold (+ fill of trimmed samples)
            cur = [e for e in edges if e.get("role") == role]
            h = self.hold.step(role, cur[0] if cur else None, ds)
            h = retrim_held(h, cur[0] if cur else None, sd, lanes)
            if h is not None:
                h["flare"] = is_flared(h.get("xg"), sd, lanes)
            if cur:
                edges[edges.index(cur[0])] = h
            elif h is not None:
                edges.append(h)
        # iter5: stripe gores need >= 3 stripes and a compact hull; plus fork gores (child diverging from parent
        # lane / split edge diverging from main edge with a growing gap)
        sg = [g for g in gore_polygons(stripes, min_stripes=3) if g["area"] <= 40.0]
        fg = self.div.step(lanes, edges)
        gores = [{"poly": np.asarray(g["poly"], float).tolist(), "n_stripes": int(g.get("n_stripes", 0)),
                  "src": g.get("src", "stripes")} for g in self.gore.step(sg + fg)]
        return lanes, edges, gores

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


KERB_CARS = os.environ.get("FBEV_KERB_CARS", "1") == "1"
KERB_MAX_PUSH_M = float(os.environ.get("FBEV_KERB_MAX_PUSH_M", "2.5"))
KERB_MARGIN_M = float(os.environ.get("FBEV_KERB_MARGIN_M", "0.0"))
KERB_CAR_LEN_M = 4.5


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
        for l in lx:
            fl = np.isfinite(l)
            if fl.sum() < 2 or np.ptp(ZS[fl]) < 8.0 or not (fl & wz).any():
                continue
            v = float(np.median(side * l[fl & wz]))
            if 0.5 < v < side * outer - 1.0:
                traffic = True; break
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


KERB_ENTER = int(os.environ.get("FBEV_KERB_ENTER", "3"))  # consecutive bounded frames before the kerb line applies
KERB_HOLD = int(os.environ.get("FBEV_KERB_HOLD", "5"))    # frames the kerb line is carried after the last bounded one
KERB_MED_N = 3                                            # kerb line = median of the last N bounded lines
KERB_HOLD_HITS = 5                                        # a hold only after >= this many bounded frames


class KerbState:
    """iter9: temporal hysteresis for kerb_behind_cars (the per-frame rule toggled with car-box flicker and made the
    edge jump by metres on a2d2a / munich): enter after KERB_ENTER consecutive bounded frames (short spurious bursts on
    munich traffic queues stay off); the line used is the
    median (offset at z = 10 m, slope, end) of the last KERB_MED_N bounded lines, each carried with the ego travel ds;
    after the last bounded frame the line is carried for KERB_HOLD frames (only if >= KERB_HOLD_HITS bounded frames),
    then dropped."""

    ZR = 10.0

    def __init__(self):
        self.s = {}

    def step(self, role, x, side, cars, ref_slope, ds=0.0, lanes=None):
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
