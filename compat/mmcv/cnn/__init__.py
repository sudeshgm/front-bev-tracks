import torch.nn as nn

from mmcv.utils import Registry

MODELS = Registry("models")


def build_conv_layer(cfg, *args, **kwargs):
    if cfg not in (None, {}):
        raise NotImplementedError(f"conv cfg {cfg} is not supported by the stub")
    return nn.Conv2d(*args, **kwargs)


def build_norm_layer(cfg, num_features, postfix=""):
    cfg = {} if cfg is None else dict(cfg)
    kind = cfg.get("type", "BN")
    if kind != "BN":
        raise NotImplementedError(f"norm {kind} is not supported by the stub")
    layer = nn.BatchNorm2d(num_features, eps=cfg.get("eps", 1e-5), momentum=cfg.get("momentum", 0.1))
    return f"bn{postfix}", layer


def build_plugin_layer(*_args, **_kwargs):
    raise NotImplementedError("plugins are not supported by the stub")
