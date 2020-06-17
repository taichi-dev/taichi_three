import taichi as ti
import taichi_glsl as ts
import numpy as np


class Ball:
    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius


@ti.data_oriented
class Render:
    EPS = 1e-3
    INF = 1e5

    def __init__(self, res=(512, 512)):
        self.res = res
        self.img = ti.Vector(3, ti.f32, self.res)
        self.balls = []

    @ti.func
    def scene_sdf(self, p):
        ret = self.INF
        for b in ti.static(self.balls):
            for I in ti.grouped(ti.ndrange(*b.pos.shape())):
                t = ts.distance(b.pos[I], p) - b.radius[I]
                ret = min(ret, t)
        return ret

    @ti.func
    def scene_grad(self, p):
        return ts.vec(
            self.scene_sdf(p + ts.vec(self.EPS, 0, 0)),
            self.scene_sdf(p + ts.vec(0, self.EPS, 0)),
            self.scene_sdf(p + ts.vec(0, 0, self.EPS)))

    @ti.kernel
    def render(self):
        for i, j in self.img:
            coor = ts.view(self.img, i, j)
            pos = ts.vec3(coor * 2.0 - 1.0, -1.0)
            dir = ts.vec3(0.0, 0.0, 1.0)
            hit = ts.vec3(0.0)
            for s in range(100):
                t = self.scene_sdf(pos)
                if t <= 0:
                    n = ts.normalize(self.scene_grad(pos) - t)
                    print(n)
                    hit = n * 0.5 + 0.5
                    break
                pos += dir * t
            self.img[i, j] = hit

    def add_ball(self, pos, radius):
        b = Ball(pos, radius)
        self.balls.append(b)
