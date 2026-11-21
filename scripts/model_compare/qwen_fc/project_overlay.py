"""Project Qwen-Drive FC-only perception back onto the KITTI front image (GPU Bot, CPU only).

Left: the original 640x480 clip frame with predicted 3D boxes (score >= --score,
official visualizer default 0.25) and the predicted BEV map (driveable surface,
road lines/edges, crosswalk, walkway) projected on the ground plane (ego z=0).
Right: bird's-eye panel, ego-forward up, with the map raster and *all* boxes,
including any the head places behind/beside the car where the cameras were black.
Geometry is the official one (geometry.build_lidar2img / box_corners, re-implemented
in numpy so this runs without torch); pixel coords are mapped from the
letterboxed 1242x(375+pad) ring image back to the squeezed 640x480 clip.
"""
import argparse, json, subprocess
from pathlib import Path
import cv2, numpy as np

DET = ("vehicle", "czone_sign", "bicycle", "generic_object", "pedestrian", "traffic_cone", "barrier")
DET_RGB = {"vehicle": (255, 138, 0), "czone_sign": (155, 81, 224), "bicycle": (32, 191, 107), "generic_object": (141, 110, 99),
           "pedestrian": (235, 59, 90), "traffic_cone": (255, 214, 10), "barrier": (127, 140, 141)}
MAP_NAMES = ("background", "driveable_surface", "road_line", "road_edge", "crosswalk", "walkway")
MAP_RGB = ((0, 0, 0), (60, 140, 255), (255, 193, 7), (225, 95, 65), (75, 180, 170), (139, 195, 74))
MAP_XB, MAP_YB = (-30.0, 30.0, 0.15), (-15.0, 15.0, 0.15)
KITTI_W, KITTI_H = 1242, 375


def bgr(rgb): return (rgb[2], rgb[1], rgb[0])


def lidar2img(K, R, t):
    r = np.linalg.inv(R); tt = t @ r.T
    rt = np.eye(4); rt[:3, :3] = r.T; rt[3, :3] = -tt
    vp = np.eye(4); vp[:3, :3] = K
    return vp @ rt.T


def box_corners(b):
    c = b[:, None, :3]; w, l, h = b[:, 3:4], b[:, 4:5], b[:, 5:6]; yaw = b[:, 6]
    co, si = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    lxy = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], float) * np.concatenate([w / 2, l / 2], -1)[:, None, :]
    lx = lxy[..., 0] * co - lxy[..., 1] * si; ly = lxy[..., 0] * si + lxy[..., 1] * co
    xy = np.stack([lx, ly], -1)
    bot = np.concatenate([xy, np.zeros((len(b), 4, 1))], -1); top = np.concatenate([xy, np.repeat(h, 4, 1)[..., None]], -1)
    return c + np.concatenate([bot, top], 1)


def project(P, pts):  # pts (...,3) lidar
    ph = np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], -1)
    q = ph @ P.T
    return q[..., :2] / np.maximum(q[..., 2:3], 1e-6), q[..., 2]


def to_disp(uv, pad, W, H):
    return np.stack([uv[..., 0] * W / KITTI_W, (uv[..., 1] - pad) * H / KITTI_H], -1)


def map_grid(m):
    m = np.asarray(m)
    if m.ndim == 3: m = m.argmax(0) if m.shape[0] == len(MAP_NAMES) else m[0]
    ny, nx = int(round((MAP_YB[1] - MAP_YB[0]) / MAP_YB[2])), int(round((MAP_XB[1] - MAP_XB[0]) / MAP_XB[2]))
    if m.shape == (nx, ny): m = m.T
    assert m.shape == (ny, nx), m.shape
    return m.astype(np.int64)  # [iy, ix]


def draw_frame(img, pred, calib, frame_meta, score):
    H, W = img.shape[:2]
    pad = frame_meta["source"]["pad_top"]
    P = lidar2img(calib["cam_intrinsic"][0], calib["sensor2lidar_rotation"][0], calib["sensor2lidar_translation"][0])
    L2E = calib["lidar2ego"]; E2L = np.linalg.inv(L2E)
    vis = img.copy(); over = img.copy()
    # map on ground plane
    m = map_grid(pred["map"])
    iy, ix = np.nonzero(m > 0)
    xs = MAP_XB[0] + MAP_XB[2] * (ix + 0.5); ys = MAP_YB[0] + MAP_YB[2] * (iy + 0.5)
    keep = xs > 2.0
    iy, ix, xs, ys = iy[keep], ix[keep], xs[keep], ys[keep]
    s = MAP_XB[2] / 2
    order = np.argsort(-xs)  # far first
    for k in order:
        quad = np.array([[xs[k] - s, ys[k] - s, 0], [xs[k] + s, ys[k] - s, 0], [xs[k] + s, ys[k] + s, 0], [xs[k] - s, ys[k] + s, 0], ])
        ql = (np.concatenate([quad, np.ones((4, 1))], 1) @ E2L.T)[:, :3]
        uv, d = project(P, ql)
        if (d <= 0.5).any(): continue
        p = to_disp(uv, pad, W, H).astype(np.int32)
        cv2.fillConvexPoly(over, p, bgr(MAP_RGB[m[iy[k], ix[k]]]))
    lines = np.isin(m, [2, 3])
    vis = cv2.addWeighted(over, 0.45, vis, 0.55, 0)
    # boxes
    sc = np.asarray(pred["scores"]).reshape(-1); lb = np.asarray(pred["labels"]).reshape(-1).astype(int)
    b = np.asarray(pred["boxes"]).reshape(len(sc), -1)[:, :9] if sc.size else np.zeros((0, 9))
    sel = np.where(sc >= score)[0]
    n_front = 0
    if len(sel):
        cs = box_corners(b[sel]); uv, d = project(P, cs)
        for j, i in enumerate(sel):
            if (d[j] <= 0.5).any(): continue
            p = to_disp(uv[j], pad, W, H)
            if not ((p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)).any(): continue
            n_front += 1
            col = bgr(DET_RGB[DET[lb[i]]]); p = np.rint(p).astype(int)
            for e in range(4):
                cv2.line(vis, tuple(p[e]), tuple(p[(e + 1) % 4]), col, 2, cv2.LINE_AA)
                cv2.line(vis, tuple(p[e + 4]), tuple(p[(e + 1) % 4 + 4]), col, 2, cv2.LINE_AA)
                cv2.line(vis, tuple(p[e]), tuple(p[e + 4]), col, 2, cv2.LINE_AA)
            cv2.putText(vis, f"{DET[lb[i]]} {sc[i]:.2f}", tuple(p[4:].min(0) + [0, -3]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    # BEV panel: 480x480, ego forward up, 40 m ahead / 20 m behind, +-30 m lateral
    S = 480; ppm = S / 60.0
    bev = np.full((S, S, 3), 30, np.uint8)
    def e2p(x, y): return np.stack([S / 2 - y * ppm, S * 2 / 3 - x * ppm], -1)
    gy, gx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    ex = MAP_XB[0] + MAP_XB[2] * (gx + 0.5); ey = MAP_YB[0] + MAP_YB[2] * (gy + 0.5)
    pp = np.rint(e2p(ex, ey)).astype(int); ok = (m > 0) & (pp[..., 0] >= 0) & (pp[..., 0] < S) & (pp[..., 1] >= 0) & (pp[..., 1] < S)
    cols = np.array([bgr(c) for c in MAP_RGB], np.uint8)
    for r in (0, 1):
        for c in (0, 1):
            yy = np.clip(pp[..., 1][ok] + r, 0, S - 1); xx = np.clip(pp[..., 0][ok] + c, 0, S - 1)
            bev[yy, xx] = cols[m[ok]]
    # front camera FOV wedge (KITTI ~81 deg)
    for sgn in (-1, 1):
        a = np.deg2rad(40.5) * sgn; end = e2p(np.array(40 * np.cos(a)), np.array(40 * np.sin(a)))
        cv2.line(bev, tuple(np.rint(e2p(np.array(1.7), np.array(0.0))).astype(int)), tuple(np.rint(end).astype(int)), (90, 90, 90), 1)
    cv2.rectangle(bev, tuple(np.rint(e2p(np.array(4.0), np.array(0.9))).astype(int)), tuple(np.rint(e2p(np.array(-1.0), np.array(-0.9))).astype(int)), (255, 255, 255), -1)
    n_behind = 0
    if len(sel):
        cl = box_corners(b[sel])[:, :4, :]
        ce = np.concatenate([cl, np.ones(cl.shape[:2] + (1,))], -1) @ L2E.T
        for j, i in enumerate(sel):
            poly = np.rint(e2p(ce[j, :, 0], ce[j, :, 1])).astype(np.int32)
            cv2.polylines(bev, [poly], True, bgr(DET_RGB[DET[lb[i]]]), 2, cv2.LINE_AA)
            if ce[j, :, 0].mean() < 0: n_behind += 1
    cv2.putText(bev, "BEV (fwd up) 1 grid=60m wide", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(bev, f"boxes>={score}: {len(sel)}  in FC view: {n_front}  behind ego: {n_behind}", (6, S - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    cv2.rectangle(vis, (0, 0), (370, 20), (0, 0, 0), -1)
    cv2.putText(vis, "Qwen-Drive-1.0 perception, FC + 5 black cams (raw)", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return np.concatenate([vis, bev], 1), {"n_boxes_over": int(len(sel)), "n_in_fc_view": n_front, "n_behind_ego": n_behind,
                                           "map_px": {MAP_NAMES[c]: int((m == c).sum()) for c in range(1, 6)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--preds", required=True); ap.add_argument("--ring-meta", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--score", type=float, default=0.25)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.clip); fps = cap.get(cv2.CAP_PROP_FPS) or 4; frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    name = Path(a.clip).stem
    preds = sorted(Path(a.preds).glob(f"{name}__*.npz"))
    outd = Path(a.out); tmp = outd / f".{name}_frames"; tmp.mkdir(parents=True, exist_ok=True)
    stats = {}
    for k, pf in enumerate(preds):
        tok = pf.stem; idx = int(tok.split("__")[1])
        rm = Path(a.ring_meta) / tok
        v, s = draw_frame(frames[idx], np.load(pf), np.load(rm / "calib.npz"), json.loads((rm / "frame.json").read_text()), a.score)
        cv2.putText(v, f"f{idx}", (590, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(str(tmp / f"{k:05d}.png"), v); stats[tok] = s
    vid = outd / f"{name}_qwen_fc.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps if len(preds) == len(frames) else 2), "-i", str(tmp / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(vid)], check=True)
    (outd / f"{name}_qwen_fc_frames.json").write_text(json.dumps(stats, indent=1))
    subprocess.run(["rm", "-rf", str(tmp)])
    print("wrote", vid, len(preds), "frames")


if __name__ == "__main__":
    main()
