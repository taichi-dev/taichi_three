import taichi as ti
import taichi_glsl as ts
from .common import EPS, INF
from .objects import Ball


class RenderOptions:
    def __init__(self, **kwargs):
        self.is_normal_map = False
        self.lambert = 0.0
        self.half_lambert = 0.5
        self.blinn_phong = 0.5
        self.phong = 0.0
        self.shineness = 12
        self.__dict__.update(kwargs)

    def __hash__(self):
        return hash(self.__dict__.values())

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


@ti.data_oriented
class SceneBase:
    def __init__(self, res=(512, 512)):
        self.res = res
        self.img = ti.Vector(3, ti.f32, self.res)
        self.light_dir = ti.Vector(3, ti.f32, ())
        self.balls = []

    def trace(self, pos, dir):
        raise NotImplementedError

    def set_light_dir(self, ldir):
        @ti.kernel
        def normalize_ldir():
            self.light_dir[None] = ts.normalize(self.light_dir[None])

        self.light_dir[None] = ldir
        normalize_ldir()

    def render(self):
        self.render_opt(RenderOptions())

    @ti.kernel
    def render_opt(self, opt: ti.template()):
        for i, j in self.img:
            coor = ts.view(self.img, i, j)
            orig = ts.vec3(coor * 2.0 - 1.0, -1.0)
            dir = ts.vec3(0.0, 0.0, 1.0)
            pos, normal = self.trace(orig, dir)
            light_dir = self.light_dir[None]

            shineness = opt.shineness
            half_lambert = ts.dot(normal, light_dir) * 0.5 + 0.5
            lambert = max(0, ts.dot(normal, light_dir))
            blinn_phong = ts.dot(normal, ts.mix(light_dir, -dir, 0.5))
            blinn_phong = pow(max(blinn_phong, 0), shineness)
            refl_dir = ts.reflect(light_dir, normal)
            phong = -ts.dot(normal, refl_dir)
            phong = pow(max(phong, 0), shineness)

            strength = 0.0
            if ti.static(opt.lambert != 0.0):
                strength += lambert * opt.lambert
            if ti.static(opt.half_lambert != 0.0):
                strength += half_lambert * opt.half_lambert
            if ti.static(opt.blinn_phong != 0.0):
                strength += blinn_phong * opt.blinn_phong
            if ti.static(opt.phong != 0.0):
                strength += phong * opt.phong
            color = ts.vec3(strength)

            if ti.static(opt.is_normal_map):
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
