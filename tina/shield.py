import taichi as ti
import numpy as np
import functools
import pickle


def _getstate(self):
    return self._pickable_decl, self.to_numpy()


def _Expr_setstate(self, state):
    (a, b), data = state

    field = ti.field(*a, **b)
    op = ti.impl.pytaichi.global_vars.pop()
    assert op is field, (op, field)
    self.__init__(field.ptr)
    ti.impl.pytaichi.global_vars.append(self)

    @ti.materialize_callback
    def restore_state():
        self.from_numpy(data)


def _Matrix_setstate(self, state):
    (a, b), data = state

    olen = len(ti.impl.pytaichi.global_vars)
    field = ti.Matrix.field(*a, **b)
    self.__init__([[field(i, j) for j in range(field.m)] for i in range(field.n)])

    @ti.materialize_callback
    def restore_state():
        self.from_numpy(data)


ti.Expr.__getstate__ = _getstate
ti.Matrix.__getstate__ = _getstate
ti.Expr.__setstate__ = _Expr_setstate
ti.Matrix.__setstate__ = _Matrix_setstate


def _mock(foo):
    @functools.wraps(foo)
    def wrapped(*a, **b):
        ret = foo(*a, **b)
        if foo == ti.Matrix._Vector_field:
            a = (a[0], 1) + a[1:]
        ret._pickable_decl = a, b
        return ret

    return wrapped


ti.field = _mock(ti.field)
ti.Vector.field = _mock(ti.Vector.field)
ti.Matrix.field = _mock(ti.Matrix.field)


print('[Tina] Taichi fields pickle hacked')


__all__ = []
