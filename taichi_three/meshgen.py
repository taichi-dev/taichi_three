import numpy as np
import taichi as ti
import math


def _pre(x):
    if not isinstance(x):
        x = ti.Vector(x)
    return x


def _ser(foo):
    def wrapped(self, *args, **kwargs):
        foo(self, *args, **kwargs)
        return self

    return wrapped


def _mparg(foo):
    def wrapped(self, *args):
        if len(args) > 1:
            return [foo(self, x) for x in args]
        else:
            return foo(self, args[0])

    return wrapped


class MeshGen:
    def __init__(self):
        self.v = []
        self.f = []

    @_ser
    def quad(self, a, b, c, d):
        ret = {}
        a, b, c, d = self.add_v(a, b, c, d)
        self.add_f([a, b, c], [c, d, a])

    @_ser
    def tri(self, a, b, c):
        ret = {}
        a, b, c = self.add_v(a, b, c)
        self.add_f([a, b, c])

    @_mparg
    def add_v(self, v):
        ret = len(self.v)
        self.v.append(v)
        return ret

    @_mparg
    def add_f(self, f):
        ret = len(self.f)
        self.f.append(f)
        return ret

    @_ser
    def cube(self, center, dir1, dir2, dir3):
        center = _pre(center)
        dir1 = _pre(dir1)
        dir2 = _pre(dir2)
        dir3 = _pre(dir3)


    def __getitem__(self, key):
        if key == 'v':
            return np.array(self.v)
        if key == 'f':
            return np.array(self.f)
