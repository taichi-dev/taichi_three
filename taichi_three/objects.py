import taichi as ti
import taichi_glsl as ts
from .common import EPS, INF


@ti.data_oriented
class Object:
    def make_one(self, scene):
        raise NotImplementedError


@ti.data_oriented
class ObjectRT(Object):
    @ti.func
    def calc_sdf(self, p):
        ret = INF
        for I in ti.grouped(ti.ndrange(*self.pos.shape())):
            ret = min(ret, self.make_one(I).do_calc_sdf(p))
        return ret

    @ti.func
    def intersect(self, orig, dir):
        ret, normal = INF, ts.vec3(0.0)
        for I in ti.grouped(ti.ndrange(*self.pos.shape())):
            t, n = self.make_one(I).do_intersect(orig, dir)
            if t < ret:
                ret, normal = t, n
        return ret, normal

    def do_calc_sdf(self, p):
        raise NotImplementedError

    def do_intersect(self, orig, dir):
        raise NotImplementedError


@ti.data_oriented
class Ball(ObjectRT):
    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius

    @ti.func
    def make_one(self, I):
        return Ball(self.pos[I], self.radius[I])

    @ti.func
    def do_calc_sdf(self, p):
        return ts.distance(self.pos, p) - self.radius

    @ti.func
    def do_intersect(self, orig, dir):
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


@ti.data_oriented
class ObjectGE(Object):
    @ti.func
    def render(self, scene):
        for I in ti.grouped(ti.ndrange(*self.a.shape())):
            self.make_one(I).do_render(scene)

    def do_render(self, scene):
        raise NotImplementedError


@ti.data_oriented
class Line(ObjectGE):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @ti.func
    def make_one(self, I):
        return Line(self.a[I], self.b[I])

    @ti.func
    def do_render(self, scene):
        width = 1
        A = scene.uncook_coor(self.a)
        B = scene.uncook_coor(self.b)
        A, B = min(A, B), max(A, B)
        BAN = ts.normalize(B - A)
        for X in ti.grouped(ti.ndrange((A.x - width, B.x + width),
                                       (A.y - width, B.y + width))):
            udf = abs(ts.cross(X - A, BAN))
            scene.img[int(X)] = ts.vec3(ts.smoothstep(udf, width, 0))


@ti.data_oriented
class Triangle(ObjectGE):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    @ti.func
    def make_one(self, I):
        return Triangle(self.a[I], self.b[I], self.c[I])

    def to_lines(self):
        return Line(self.b, self.c), Line(self.c, self.a), Line(self.a, self.b)

    @ti.func
    def do_render(self, scene):
        A, B, C = self.to_lines()
        A.do_render(scene)
        B.do_render(scene)
        C.do_render(scene)
