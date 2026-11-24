"""iter4: inverse-perspective BEV image (flat ground), paint mask, stripe orientation, asphalt mask.
Classical CV on the camera frame only; the GPU pipeline uses the SAM 2.1 road/paint masks instead of
asphalt_mask/paint_mask here, and the same downstream line logic."""
import numpy as np
import cv2

RES = 0.1          # m per BEV pixel
X_RANGE = (-15.0, 15.0)
Z_RANGE = (5.0, 45.0)


def bev_shape(res=RES, xr=X_RANGE, zr=Z_RANGE):
    return int(round((zr[1] - zr[0]) / res)), int(round((xr[1] - xr[0]) / res))


def xz_to_bev_px(x, z, res=RES, xr=X_RANGE, zr=Z_RANGE):
    H, W = bev_shape(res, xr, zr)
    return (np.asarray(x) - xr[0]) / res, H - 1 - (np.asarray(z) - zr[0]) / res


def bev_px_to_xz(u, v, res=RES, xr=X_RANGE, zr=Z_RANGE):
    H, W = bev_shape(res, xr, zr)
    return xr[0] + np.asarray(u) * res, zr[0] + (H - 1 - np.asarray(v)) * res


def ipm_homography(gm, res=RES, xr=X_RANGE, zr=Z_RANGE):
    """Homography image -> BEV pixels from 4 ground points (gm: bev_smooth.GroundModel)."""
    g = np.array([[xr[0] * 0.5, zr[0] + 2], [xr[1] * 0.5, zr[0] + 2], [xr[1] * 0.5, zr[1] - 5], [xr[0] * 0.5, zr[1] - 5]])
    img = gm.project(g).astype(np.float32)
    u, v = xz_to_bev_px(g[:, 0], g[:, 1], res, xr, zr)
    return cv2.getPerspectiveTransform(img, np.stack([u, v], 1).astype(np.float32))


def warp(img, Hm, res=RES, xr=X_RANGE, zr=Z_RANGE, interp=cv2.INTER_LINEAR):
    H, W = bev_shape(res, xr, zr)
    return cv2.warpPerspective(img, Hm, (W, H), flags=interp, borderValue=0)


def valid_mask(shape_img, Hm, horizon_v, res=RES, xr=X_RANGE, zr=Z_RANGE):
    m = np.zeros(shape_img[:2], np.uint8)
    m[int(horizon_v) + 6:] = 255
    return warp(m, Hm, res, xr, zr, cv2.INTER_NEAREST) > 0


def paint_mask(bev_gray, valid, width_m=0.6, res=RES, k=2.2):
    """Bright thin markings: horizontal top-hat (lines are near-vertical in BEV) OR vertical top-hat
    (chevrons / transverse stripes), thresholded at k sigma over the asphalt."""
    kw = max(3, int(round(width_m / res)) | 1)
    th = cv2.morphologyEx(bev_gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1)))
    tv = cv2.morphologyEx(bev_gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kw)))
    t = np.maximum(th, tv).astype(np.float32)
    vals = t[valid]
    if vals.size < 100:
        return np.zeros_like(valid)
    med = np.median(vals); mad = np.median(np.abs(vals - med)) + 1.0
    m = (t > med + max(18.0, k * 6 * mad)) & valid
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    return m


def stripe_orientation(bev_gray, sigma=3.0):
    """Structure-tensor orientation of the dominant edge direction (radians, 0 = along z / vertical in
    BEV) and coherence in [0,1]."""
    g = bev_gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma); jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    # gradient angle; stripe direction is perpendicular to it
    ang = 0.5 * np.arctan2(2 * jxy, jxx - jyy)          # gradient direction from +u axis
    stripe = ang + np.pi / 2                             # stripe direction from +u axis
    # convert to angle from the vertical (z) axis, folded to [0, pi/2]
    from_vert = np.abs(((stripe - np.pi / 2) + np.pi / 2) % np.pi - np.pi / 2)
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    return from_vert, coh


def asphalt_mask(bev_bgr, valid, seed_xz=((0.0, 10.0), (-0.8, 12.0), (0.8, 12.0), (0.0, 14.0)), res=RES, tol=2.5, paint=None):
    """Drivable asphalt by colour similarity to seed patches in front of the ego (Lab, Mahalanobis),
    closed over markings, restricted to the component connected to the seeds."""
    lab = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = []
    for x, z in seed_xz:
        u, v = xz_to_bev_px(x, z, res)
        u, v = int(u), int(v)
        p = lab[max(0, v - 8):v + 8, max(0, u - 8):u + 8].reshape(-1, 3)
        samples.append(p)
    S = np.concatenate(samples, 0)
    mu = S.mean(0); C = np.cov(S.T) + np.eye(3) * 16.0
    Ci = np.linalg.inv(C)
    d = lab.reshape(-1, 3) - mu
    m2 = np.einsum("ij,jk,ik->i", d, Ci, d).reshape(lab.shape[:2])
    g = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mu_g = cv2.blur(g, (15, 15)); sd = np.sqrt(np.maximum(cv2.blur(g * g, (15, 15)) - mu_g ** 2, 0))
    seed_sd = np.median([sd[int(xz_to_bev_px(x, z, res)[1]), int(xz_to_bev_px(x, z, res)[0])] for x, z in seed_xz])
    smooth_tex = sd < max(6.0, 2.5 * seed_sd)      # asphalt is smooth; verges/grass/gravel are textured
    # markings are drivable surface and must not split the asphalt into components: texture is measured
    # with the paint pixels replaced by the asphalt mean
    if paint is not None:
        g2 = g.copy(); g2[paint] = float(np.median(g[valid])) if valid.any() else 0.0
        mu_g = cv2.blur(g2, (15, 15)); sd = np.sqrt(np.maximum(cv2.blur(g2 * g2, (15, 15)) - mu_g ** 2, 0))
        smooth_tex = sd < max(6.0, 2.5 * seed_sd)
    m = (((m2 < tol * tol) & smooth_tex) | (paint if paint is not None else False)) & valid
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab_cc = cv2.connectedComponents(m)
    keep = set()
    for x, z in seed_xz:
        u, v = xz_to_bev_px(x, z, res)
        l = lab_cc[int(v), int(u)]
        if l > 0:
            keep.add(l)
    return np.isin(lab_cc, list(keep)) if keep else m > 0
