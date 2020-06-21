import taichi as ti
import taichi_glsl as ts
from .common import EPS, INF
from .object_rt import Ball
from .scene import *
import math


@ti.data_oriented
class SceneRTBase(SceneBase):
    def __init__(self, res=None):
        super(SceneRTBase, self).__init__(res)
        self.balls = []

    def trace(self, pos, dir):
        raise NotImplementedError

    @ti.func
    def color_at(self, coor):
        orig, dir = self.camera.generate(coor)

        pos, normal = self.trace(orig, dir)
        light_dir = self.light_dir[None]

        color = self.opt.render_func(pos, normal, dir, light_dir)
        color = self.opt.pre_process(color)
        return color

    @ti.kernel
    def do_render(self):
        for I in ti.grouped(self.img):
            coor = self.cook_coor(I)
            color = self.color_at(coor)
            self.img[I] = color

    def add_ball(self, pos, radius):
        b = Ball(pos, radius)
        self.balls.append(b)


@ti.data_oriented
class SceneRT(SceneRTBase):
    @ti.func
    def intersect(self, orig, dir):
        ret, normal = INF, ts.vec3(0.0)
        for b in ti.static(self.balls):
            t, n = b.intersect(orig, dir)
            if t < ret:
                ret, normal = t, n
        return ret, normal

    @ti.func
    def trace(self, orig, dir):
        depth, normal = self.intersect(orig, dir)
        pos = orig + dir * depth
        return pos, normal


@ti.data_oriented
class SceneSDF(SceneRT):
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
    def trace(self, orig, dir):
        pos = orig
        color = ts.vec3(0.0)
        normal = ts.vec3(0.0)
        for s in range(100):
            t = self.calc_sdf(pos)
            if t <= 0:
                normal = ts.normalize(self.calc_grad(pos) - t)
                break
            pos += dir * t
        return pos, normal
