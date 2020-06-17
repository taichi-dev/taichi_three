import taichi as ti
import taichi_glsl as ts
import numpy as np

EPS = 1e-3
INF = 1e5


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
class SceneBase:
    def __init__(self, res=(512, 512)):
        self.res = res
        self.img = ti.Vector(3, ti.f32, self.res)
        self.balls = []

    def trace(self, pos, dir):
        raise NotImplementedError

    @ti.kernel
    def render(self):
        for i, j in self.img:
            coor = ts.view(self.img, i, j)
            pos = ts.vec3(coor * 2.0 - 1.0, -1.0)
            dir = ts.vec3(0.0, 0.0, 1.0)
            normal = self.trace(pos, dir)
            color = normal * 0.5 + 0.5
            self.img[i, j] = color

    def add_ball(self, pos, radius):
        b = Ball(pos, radius)
        self.balls.append(b)


@ti.data_oriented
class Scene(SceneBase):
    @ti.func
    def intersect(self, orig, dir):
        ret, normal = INF, ts.vec3(0.0)
        for b in ti.static(self.balls):
            t, n = b.intersect(orig, dir)
            if t < ret:
                ret, normal = t, n
        return ret, normal

    @ti.func
    def trace(self, pos, dir):
        depth, normal = self.intersect(pos, dir)
        return normal


@ti.data_oriented
class SceneSDF(SceneBase):
    @ti.func
    def calc_sdf(self, p):
        ret = INF
        for b in ti.static(self.balls):
            ret = min(ret, b.calc_sdf(p))
        return ret

    @ti.func
    def calc_grad(self, p):
        return ts.vec(
            self.calc_sdf(p + ts.vec(EPS, 0, 0)),
            self.calc_sdf(p + ts.vec(0, EPS, 0)),
            self.calc_sdf(p + ts.vec(0, 0, EPS)))

    @ti.func
    def trace(self, pos, dir):
        color = ts.vec3(0.0)
        normal = ts.vec3(0.0)
        for s in range(100):
            t = self.calc_sdf(pos)
            if t <= 0:
                normal = ts.normalize(self.calc_grad(pos) - t)
                break
            pos += dir * t
        return normal
