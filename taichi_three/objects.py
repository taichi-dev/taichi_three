import taichi as ti
import taichi_glsl as ts
from .common import EPS, INF


@ti.data_oriented
class Ball:
    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius

    @ti.func
    def make_one(self, I):
        return Ball(self.pos[I], self.radius[I])

    @ti.func
    def _calc_sdf(self, p):
        return ts.distance(self.pos, p) - self.radius

    @ti.func
    def calc_sdf(self, p):
        ret = INF
        for I in ti.grouped(ti.ndrange(*self.pos.shape())):
            ret = min(ret, self.make_one(I)._calc_sdf(p))
        return ret

    @ti.func
    def _intersect(self, orig, dir):
        op = self.pos - orig
        b = op.dot(dir)
        det = b ** 2 - op.norm_sqr() + self.radius ** 2
        ret = INF
        if det > 0.0:
            det = ti.sqrt(det)
            t = b - det
            if t > EPS:
                ret = t
            else:
                t = b + det
                if t > EPS:
                    ret = t
        return ret, ts.normalize(dir * ret - op)

    @ti.func
    def intersect(self, orig, dir):
        ret, normal = INF, ts.vec3(0.0)
        for I in ti.grouped(ti.ndrange(*self.pos.shape())):
            t, n = self.make_one(I)._intersect(orig, dir)
            if t < ret:
                ret, normal = t, n
        return ret, normal


@ti.data_oriented
class Triangle:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    @ti.func
    def make_one(self, I):
        return Triangle(self.a[I], self.b[I], self.c[I])

    @ti.func
    def _render(self, scene):
        I = scene.uncook_coor(self.a)
        scene.img[int(I)] = ts.vec3(1.0, 0.0, 0.0)
        I = scene.uncook_coor(self.b)
        scene.img[int(I)] = ts.vec3(0.0, 1.0, 0.0)
        I = scene.uncook_coor(self.c)
        scene.img[int(I)] = ts.vec3(0.0, 0.0, 1.0)

    @ti.func
    def render(self, scene):
        for I in ti.grouped(ti.ndrange(*self.a.shape())):
            self.make_one(I)._render(scene)
