"""Synthetic test of the patched decode_boxes on both reported layouts (needs torch)."""
import sys, os, shutil, tempfile, importlib.util, torch
here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fbtchk/scripts/overlay.py"
tmp = tempfile.mkdtemp(); dst = os.path.join(tmp, "overlay_patched.py"); shutil.copy(src, dst)
sys.path.insert(0, here); import decode_fix
assert decode_fix.patch(dst) == 0 and decode_fix.patch(dst) == 0          # idempotent
sys.path.insert(0, os.path.dirname(os.path.dirname(src)))
spec = importlib.util.spec_from_file_location("ov", dst); ov = importlib.util.module_from_spec(spec)
sys.argv = ["x"]; spec.loader.exec_module(ov)
torch.manual_seed(0)
anchors = [torch.tensor([[12., 16], [19, 36], [40, 28]]), torch.tensor([[36., 75], [76, 55], [72, 146]]), torch.tensor([[142., 110], [192, 243], [459, 401]])]
def head(layout):
    out = []
    for ny in (80, 40, 20):
        base = torch.full((3, 85, ny, ny), -8.0)
        base[1, 0:4, 10, 12] = 0.0; base[1, 4, 10, 12] = 8.0; base[1, 5 + 2, 10, 12] = 8.0             # one confident car at level-local cell
        if layout == "255":
            out.append(base.reshape(1, 255, ny, ny))
        elif layout == "3x1":                                              # the iter3 failure: extra singleton
            out.append(base.permute(0, 2, 3, 1).reshape(3, 1, ny, ny, 85))
        elif layout == "1x3xyx85":
            out.append(base.permute(0, 2, 3, 1).unsqueeze(0))
    return out
res = {}
for lay in ("255", "3x1", "1x3xyx85"):
    from types import SimpleNamespace
    b = ov.decode_boxes((head(lay), anchors), SimpleNamespace(width=640, height=480))
    n = 0 if isinstance(b, list) and not b else (sum(len(x) for x in b) if isinstance(b, list) else len(b))
    res[lay] = n
assert len(set(res.values())) == 1 and list(res.values())[0] >= 1, res
print("DECODE FIX OK", res)
