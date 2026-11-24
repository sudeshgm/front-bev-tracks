# Simulate the NumPy 1.x API surface for the iter5b crash: no np.trapezoid, only np.trapz.
import numpy as _np
_t = _np.trapezoid
del _np.trapezoid
_np.trapz = _t
if "trapezoid" in getattr(_np, "__all__", []):  # NumPy 1.x does not export it
    _np.__all__ = [n for n in _np.__all__ if n != "trapezoid"]
print("NP1_SIM active: hasattr(np,'trapezoid') =", hasattr(_np, "trapezoid"), file=__import__("sys").stderr, flush=True)
