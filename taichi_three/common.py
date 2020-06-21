import taichi as ti


class TaichiClass:
    is_taichi_class = True

    def __init__(self, *entries):
        self.entries = entries

    @classmethod
    def var(cls):
        var_list = cls._var()
        if not isinstance(var_list, (list, tuple)):
            var_list = [var_list]
        cls(*var_list)

    @classmethod
    def _var(cls):
        raise NotImplementedError

    @ti.func
    def _subscript(self, I):
        return self.__class__(*(e[I] for e in self.entries))

    def subscript(self, I):
        return self._subscript(I)

    @ti.func
    def loop_range(self):
        return self.entries[0]
