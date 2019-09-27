from typing import TypeVar, MutableMapping


X = TypeVar("X")

class Registry(MutableMapping[str, X]):
    """Registers function or classes as a dict.
    """

    def __init__(self):
        # super().__init__()
        self.data = {}
        self._information = {}

    # def register(self, obj: X, info: Optional[Dict[Any, Any]] = None) -> X:
    def register(self, obj, info=None):
        """Decorator method for registering functions/classes
        The info variable can be filled up using the register_with_info
        decorator instead of this one.
        """
        name = getattr(obj, "__name__", obj.__class__.__name__)
        self.register_name(name, obj, info)
        return obj

    # def register_name(self, name: str, obj: X, info: Optional[Dict[Any, Any]] = None) -> None:
    def register_name(self, name, obj, info):
        """Register an object with a provided name
        """
        self[name] = obj ## Cross check implementation
        if info is not None:
            assert isinstance(info, dict)
            self._information[name] = info

    def unregister(self, name):
        """Remove a previously-registered function or class, e.g. so you can
        re-register it in a Jupyter notebook.
        """
        if name in self:
            del self[name]

    # def register_with_info(self, **info: Any) -> Callable[[X], X]:
    #     """Decorator for registering a function and information about it
    #     """
    #     return functools.partial(self.register, info=info)

    def get_info(self, name):
        if name not in self:
            raise ValueError(f'"{name}" is not registered.')
        return self._information.setdefault(name, {})


    def __getitem__(self, key): ## Try resetting! Why??
        return self.data[key]

    def __setitem__(self, key, value): ## Try resetting! Why??
        self.data[key] = value

    def __delitem__(self, key): ## Why??
        del self.data[key]

    def __iter__(self): ## Why??
        return iter(self.data)

    def __len__(self): 
        return len(self.data)



    # def __getitem__(self, key: str) -> X:
    #     return self.data[key]

    # def __setitem__(self, key: str, value: X) -> None:
    #     self.data[key] = value

    # def __delitem__(self, key: str) -> None:
    #     del self.data[key]

    # def __iter__(self) -> Iterator[str]:
    #     return iter(self.data)

    # def __len__(self) -> int:
    #     return len(self.data)