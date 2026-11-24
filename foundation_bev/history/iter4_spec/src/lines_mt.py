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
import numpy as np
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


class StageLines:
    """Per-clip lane / edge / gore tracking on BEV masks (road: drivable, paint: markings)."""

    def __init__(self):
        self.lanes = MultiLineTracker("lane")
        self.edges = {r: MultiLineTracker("edge", n_birth=3, n_death=4, gate_max=2.0, alpha=0.3)
                      for r in ("main_L", "main_R", "split_in", "split_out")}
        self.gore = GoreTracker()
        self.n_split_frames = 0

    def step(self, k, road_bev, paint_bev, valid_bev, res):
        comps = components(paint_bev, res)
        chains, stripes = extract_chains(comps)
        lanes = [_legacy(r) for r in self.lanes.step(k, [smooth_chain(c) for c in chains])]
        road_f = cv2.morphologyEx((road_bev | paint_bev).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        ed = edge_chains(road_f, res, valid=cv2.erode(valid_bev.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        if np.isfinite(ed["split_in"]).any():
            self.n_split_frames += 1
        edges = []
        for role, tr in self.edges.items():
            got = tr.step(k, [ed[role]])
            live = [r for r in got if not r["held"]]
            got = live if live else got[:1]  # one boundary per role: drop a coasting duplicate after a re-birth
            for r in got:
                side = -1 if role == "main_L" else (1 if role == "main_R" else None)
                edges.append(_legacy(r, side=side, kind="edge", extra={"role": role, "split": role.startswith("split")}))
        gores = [{"poly": np.asarray(g["poly"], float).tolist(), "n_stripes": int(g["n_stripes"])}
                 for g in self.gore.step(gore_polygons(stripes))]
        return lanes, edges, gores

    def events(self):
        return self.lanes.events + [e for t in self.edges.values() for e in t.events]
