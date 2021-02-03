import taichi as ti
import numpy as np

from .hacker import *


MAX = 2**20


def V(*xs):
    return ti.Vector(xs)


def V23(xy, z):
    return V(xy.x, xy.y, z)


def V34(xyz, w):
    return V(xyz.x, xyz.y, xyz.z, w)


def V2(x):
    if isinstance(x, ti.Matrix):
        return x
    else:
        return V(x, x)


def V3(x):
    if isinstance(x, ti.Matrix):
        return x
    else:
        return V(x, x, x)


def Vavg(u):
    if isinstance(u, ti.Matrix):
        return u.sum() / len(u.entries)
    else:
        return u


def Vall(u):
    if isinstance(u, ti.Matrix):
        return u.all()
    else:
        return u


def Vlen2(u):
    if isinstance(u, ti.Matrix):
        return u.norm_sqr()
    else:
        return u**2


def Vlen(u):
    if isinstance(u, ti.Matrix):
        return u.norm()
    else:
        return u2


def Vany(u):
    if isinstance(u, ti.Matrix):
        return u.any()
    else:
        return u


def U3(i):
    return ti.Vector.unit(3, i)


def U2(i):
    return ti.Vector.unit(2, i)


ti.Matrix.xy = property(lambda v: V(v.x, v.y))
ti.Matrix.Yx = property(lambda v: V(-v.y, v.x))
ti.Matrix.xZy = property(lambda v: V(v.x, -v.z, v.y))
ti.Matrix.xyz = property(lambda v: V(v.x, v.y, v.z))


@ti.pyfunc
def Vprod(w):
    v = tovector(w)
    if ti.static(not v.entries):
        return 1
    x = v.entries[0]
    if ti.static(len(v.entries) > 1):
        for y in ti.static(v.entries[1:]):
            x *= y
    return x


def totuple(x):
    if x is None:
        x = []
    if isinstance(x, ti.Matrix):
        x = x.entries
    if isinstance(x, list):
        x = tuple(x)
    if not isinstance(x, tuple):
        x = [x]
    if isinstance(x, tuple) and len(x) and x[0] is None:
        x = []
    return tuple(x)


def tovector(x):
    return ti.Vector(totuple(x))


def vconcat(*xs):
    res = []
    for x in xs:
        if isinstance(x, ti.Matrix):
            res.extend(x.entries)
        else:
            res.append(x)
    return ti.Vector(res)


@ti.pyfunc
def clamp(x, xmin=0, xmax=1):
    return min(xmax, max(xmin, x))


@ti.pyfunc
def ifloor(x):
    return int(ti.floor(x))


@ti.pyfunc
def iceil(x):
    return int(ti.ceil(x))


@ti.func
def bilerp(f: ti.template(), pos):
    p = float(pos)
    I = ifloor(p)
    x = p - I
    y = 1 - x
    return (f[I + V(1, 1)] * x[0] * x[1] +
            f[I + V(1, 0)] * x[0] * y[1] +
            f[I + V(0, 0)] * y[0] * y[1] +
            f[I + V(0, 1)] * y[0] * x[1])

@ti.func
def trilerp(f: ti.template(), pos):
    p = float(pos)
    I = ifloor(p)
    w0 = p - I
    w1 = 1 - w0

    c00 = f[I + V(0,0,0)] * w1.x + f[I + V(1,0,0)] * w0.x
    c01 = f[I + V(0,0,1)] * w1.x + f[I + V(1,0,1)] * w0.x
    c10 = f[I + V(0,1,0)] * w1.x + f[I + V(1,1,0)] * w0.x
    c11 = f[I + V(0,1,1)] * w1.x + f[I + V(1,1,1)] * w0.x

    c0 = c00 * w1.y + c10 * w0.y
    c1 = c01 * w1.y + c11 * w0.y

    return c0 * w1.z + c1 * w0.z


@ti.func
def mapply(mat, pos, wei):
    res = ti.Vector([mat[i, 3] for i in range(3)]) * wei
    for i, j in ti.static(ti.ndrange(3, 3)):
        res[i] += mat[i, j] * pos[j]
    rew = mat[3, 3] * wei
    for i in ti.static(range(3)):
        rew += mat[3, i] * pos[i]
    return res, rew


@ti.func
def mapply_pos(mat, pos):
    res, rew = mapply(mat, pos, 1)
    return res / rew


@ti.func
def mapply_dir(mat, dir):
    res, rew = mapply(mat, dir, 0)
    return res


@ti.func
def linear_part(mat):
    return ti.Matrix([[mat[i, j] for j in range(3)] for i in range(3)])


@ti.pyfunc
def reflect(I, N):
    return I - 2 * N.dot(I) * N


@ti.pyfunc
def refract(I, N, ior):
    has_r, T = 0, I
    NoI = N.dot(I)
    discr = 1 - ior**2 * (1 - NoI**2)
    if discr > 0:
        has_r = 1
        T = (ior * (I - N * NoI) - N * ti.sqrt(discr)).normalized()
    else:
        T *= 0
    return has_r, T


@ti.pyfunc
def smoothstep(x, a, b):
    t = clamp((x - a) / (b - a))
    return t * t * (3 - 2 * t)


@ti.pyfunc
def lerp(fac, src, dst):
    return src * (1 - fac) + dst * fac


@ti.pyfunc
def smoothlerp(fac, src, dst):
    if fac != 0 and fac != 1:
        fac = smoothstep(fac, 0, 1)
        fac = src * (1 - fac) + dst * fac
    return fac


@ti.pyfunc
def unlerp(val, src, dst):
    return (val - src) / (dst - src)


@ti.func
def list_subscript(a, i):
    ret = sum(a) * 0
    for j in ti.static(range(len(a))):
        if i == j:
            ret = a[j]
    return ret


@ti.func
def isnan(x):
    return not (x >= 0 or x <= 0)


def ranprint(*args, **kwargs):
    @ti.func
    def func(r):
        if ti.random() < r:
            print(*args)

    func(kwargs.get('r', 1e-3))


class namespace(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from None


import tina
