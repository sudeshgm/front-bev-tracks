class Registry:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self._module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        def register(cls):
            key = name or cls.__name__
            self._module_dict[key] = cls
            return cls

        if module is not None:
            return register(module)
        return register

    def get(self, key):
        found = self._module_dict.get(key)
        if found is None and self.parent is not None:
            found = self.parent.get(key)
        if found is None:
            raise KeyError(f"{key} is not in the {self.name} registry")
        return found

    def build(self, cfg, default_args=None):
        cfg = cfg.copy()
        if default_args:
            for key, value in default_args.items():
                cfg.setdefault(key, value)
        obj_type = cfg.pop("type")
        obj_cls = obj_type if not isinstance(obj_type, str) else self.get(obj_type)
        return obj_cls(**cfg)
