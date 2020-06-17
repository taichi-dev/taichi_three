import taichi as ti
import taichi_glsl as ts
from .common import EPS, INF
from .objects import Ball


class RenderOptions:
    def __init__(self, **kwargs):
        self.is_normal_map = False
        self.lambert = 0.4
        self.half_lambert = 0.4
        self.blinn_phong = 0.4
        self.phong = 0.0
        self.shineness = 12
        self.render_func = None
        self.__dict__.update(kwargs)


@ti.data_oriented
class SceneBase:
    def __init__(self, res=(512, 512)):
        self.res = res
        self.img = ti.Vector(3, ti.f32, self.res)
        self.light_dir = ti.Vector(3, ti.f32, ())
        self.opt = RenderOptions()
        self.balls = []

    def trace(self, pos, dir):
        raise NotImplementedError

    def set_light_dir(self, ldir):
        import math
        norm = math.sqrt(sum(x**2 for x in ldir))
        ldir = [x / norm for x in ldir]
        self.light_dir[None] = ldir

    @ti.func
    def render_func(self, pos, normal, dir, light_dir):
        color = ts.vec3(0.0)

        if ti.static(self.opt.is_normal_map):
            color = normal * 0.5 + 0.5

        elif ti.static(self.opt.render_func is not None):
            color = self.opt.render_func(
                    pos, normal, dir, light_dir)

        else:
            shineness = self.opt.shineness
            half_lambert = ts.dot(normal, light_dir) * 0.5 + 0.5
            lambert = max(0, ts.dot(normal, light_dir))
            blinn_phong = ts.dot(normal, ts.mix(light_dir, -dir, 0.5))
            blinn_phong = pow(max(blinn_phong, 0), shineness)
            refl_dir = ts.reflect(light_dir, normal)
            phong = -ts.dot(normal, refl_dir)
            phong = pow(max(phong, 0), shineness)

            strength = 0.0
            if ti.static(self.opt.lambert != 0.0):
                strength += lambert * self.opt.lambert
            if ti.static(self.opt.half_lambert != 0.0):
                strength += half_lambert * self.opt.half_lambert
            if ti.static(self.opt.blinn_phong != 0.0):
                strength += blinn_phong * self.opt.blinn_phong
            if ti.static(self.opt.phong != 0.0):
                strength += phong * self.opt.phong
            color = ts.vec3(strength)

        return color

    @ti.kernel
    def render(self):
        for I in ti.grouped(self.img):
            scale = ti.static(2 / min(*self.img.shape()))
            coor = (I - ts.vec2(*self.img.shape()) / 2) * scale

            orig = ts.vec3(coor, -1.0)
            dir = ts.vec3(0.0, 0.0, 1.0)
            pos, normal = self.trace(orig, dir)
            light_dir = self.light_dir[None]

            color = self.render_func(pos, normal, dir, light_dir)
            self.img[I] = color

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
    def trace(self, orig, dir):
        depth, normal = self.intersect(orig, dir)
        pos = orig + dir * depth
        return pos, normal


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
