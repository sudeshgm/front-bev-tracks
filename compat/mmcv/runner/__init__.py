import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        pass


class Sequential(BaseModule):
    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        if len(args) == 1 and isinstance(args[0], dict):
            modules = args[0].items()
        else:
            modules = ((str(index), module) for index, module in enumerate(args))
        for name, module in modules:
            self.add_module(name, module)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


def auto_fp16(*_args, **_kwargs):
    def decorate(function):
        return function

    return decorate


def force_fp32(*_args, **_kwargs):
    def decorate(function):
        return function

    return decorate
