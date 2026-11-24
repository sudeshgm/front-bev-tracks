"""Patch fbt_c2b386b scripts/overlay.py decode_boxes for the YOLOPv2 TorchScript head layout.

iter3 GPU run: IndexError at decode_boxes (`center[keep]`), mask [3,80,80] vs tensor [3,1,80,80]: the
exported head returns per-level tensors whose layout differs from the [1,255,ny,nx] the code assumes
(here an extra singleton axis survives into center/size). The fix normalises every level to
scores[3, ny, nx, 85] whatever the export layout ([1,255,ny,nx], [1,3,85,ny,nx], [1,3,ny,nx,85],
[3,1,ny,nx,85], ...), the anchors to [3,1,1,2], and flattens before boolean indexing.
Usage: python decode_fix.py path/to/scripts/overlay.py   (idempotent)"""
import re, sys

NEW = '''def _yolo_level_scores(feat):
    """Any YOLOPv2 head layout -> sigmoid scores [3, ny, nx, 85]."""
    t = feat
    while t.dim() > 4 and t.shape[0] == 1:
        t = t[0]
    if t.dim() == 5:  # e.g. [3, 1, ny, nx, 85]
        t = t.reshape(t.shape[0], *[s for s in t.shape[1:] if s != 1]) if 1 in t.shape[1:-1] else t[0]
    if t.dim() == 4 and t.shape[0] == 1 and t.shape[1] == 255:      # [1, 255, ny, nx]
        ny, nx = t.shape[-2:]
        t = t.view(3, 85, ny, nx).permute(0, 2, 3, 1)
    elif t.dim() == 3 and t.shape[0] == 255:                         # [255, ny, nx]
        ny, nx = t.shape[-2:]
        t = t.view(3, 85, ny, nx).permute(0, 2, 3, 1)
    elif t.dim() == 4 and t.shape[0] == 3 and t.shape[1] == 85:      # [3, 85, ny, nx]
        t = t.permute(0, 2, 3, 1)
    elif t.dim() == 4 and t.shape[0] == 3 and t.shape[-1] == 85:     # [3, ny, nx, 85]
        pass
    else:
        raise ValueError(f"unexpected YOLOPv2 head shape {tuple(feat.shape)}")
    return t.sigmoid()


def decode_boxes(pred, camera, conf=0.4):
    feats, anchors = pred
    strides = [8, 16, 32]
    candidates = []
    for index, feat in enumerate(feats):
        scores = _yolo_level_scores(feat)
        _, ny, nx, _ = scores.shape
        anchor = anchors[index].reshape(3, 1, 1, 2).to(scores.dtype)
        gy, gx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
        grid = torch.stack((gx, gy), -1).float().view(1, ny, nx, 2)
        center = ((scores[..., 0:2] * 2 - 0.5 + grid) * strides[index]).reshape(-1, 2)
        size = ((scores[..., 2:4] * 2) ** 2 * anchor).reshape(-1, 2)
        score, _kind = scores[..., 5:].max(-1)
        score = (score * scores[..., 4]).reshape(-1)
        keep = score > conf
        if keep.any():
            candidates.append(torch.cat([center[keep] - size[keep] / 2, center[keep] + size[keep] / 2, score[keep, None]], 1))
    if not candidates:
        return []
'''


def patch(path):
    s = open(path).read()
    if "_yolo_level_scores" in s:
        print("decode_fix: already applied"); return 0
    m = re.search(r"def decode_boxes\(pred, camera, conf=0\.4\):\n.*?    if not candidates:\n        return \[\]\n", s, re.S)
    if not m:
        print("decode_fix: decode_boxes block not found"); return 2
    open(path, "w").write(s[:m.start()] + NEW + s[m.end():])
    print("decode_fix: patched", path); return 0


if __name__ == "__main__":
    sys.exit(patch(sys.argv[1]))
