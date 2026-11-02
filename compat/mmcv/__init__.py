"""Just enough of mmcv 1.6 for Anchor3DLane. No compiled ops."""

__version__ = "1.6.2"


def load(*_args, **_kwargs):
    raise NotImplementedError("mmcv.load is not available in the stub")


def dump(*_args, **_kwargs):
    raise NotImplementedError("mmcv.dump is not available in the stub")
