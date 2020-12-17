import numpy as np
import taichi as ti
import functools


class AutoInit:
    def init(self):
        if not hasattr(self, '_AutoInit_had_init'):
            self._init()
            self._AutoInit_had_init = True

    def _init(self):
        raise NotImplementedError


class subscriptable(property):
    def __init__(self, func):
        self.func = func

        @functools.wraps(func)
        def accessor(this):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                return func(this, *args, **kwargs)

            @functools.wraps(func)
            def subscript(*indices):
                if len(indices) == 1 and indices[0] is None:
                    indices = []
                return func(this, *indices)

            wrapped.subscript = subscript
            wrapped.is_taichi_class = True

            return wrapped

        super().__init__(accessor)

class dummy_expression:
    is_taichi_class = True

    def __getattr__(self, key):
        def wrapped(*args, **kwargs):
            return self
        wrapped.__name__ = key
        return wrapped

_old_begin_frontend_if = ti.begin_frontend_if

def _begin_frontend_if(cond):
    if isinstance(cond, dummy_expression):
        cond = 0
    return _old_begin_frontend_if(cond)

ti.begin_frontend_if = _begin_frontend_if
