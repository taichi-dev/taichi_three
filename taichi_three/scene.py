import taichi as ti
import taichi_glsl as ts
from .common import EPS, INF
from .objects import Ball


class RenderOptions:
    def __init__(self, **kwargs):
        self.is_normal_map = False
        self.lambert = 0.58
        self.half_lambert = 0.04
        self.blinn_phong = 0.28
        self.phong = 0.0
        self.shineness = 10
        self.__dict__.update(kwargs)

    @ti.func
    def render_func(self, pos, normal, dir, light_dir):
        color = ts.vec3(0.0)

        shineness = self.shineness
        half_lambert = ts.dot(normal, light_dir) * 0.5 + 0.5
        lambert = max(0, ts.dot(normal, light_dir))
        blinn_phong = ts.dot(normal, ts.mix(light_dir, -dir, 0.5))
        blinn_phong = pow(max(blinn_phong, 0), shineness)
        refl_dir = ts.reflect(light_dir, normal)
        phong = -ts.dot(normal, refl_dir)
        phong = pow(max(phong, 0), shineness)

        strength = 0.0
        if ti.static(self.lambert != 0.0):
            strength += lambert * self.lambert
        if ti.static(self.half_lambert != 0.0):
            strength += half_lambert * self.half_lambert
        if ti.static(self.blinn_phong != 0.0):
            strength += blinn_phong * self.blinn_phong
        if ti.static(self.phong != 0.0):
            strength += phong * self.phong
        color = ts.vec3(strength)

        if ti.static(self.is_normal_map):
            color = normal * 0.5 + 0.5

        return color

    @ti.func
    def pre_process(self, color):
        blue = ts.vec3(0.0, 0.01, 0.05)
        orange = ts.vec3(1.19, 1.04, 0.98)
        return ti.sqrt(ts.mix(blue, orange, color))


@ti.data_oriented
class SceneBase:
    def __init__(self, res=(512, 512)):
        self.res = res
        self.img = ti.Vector(3, ti.f32, self.res)
        self.light_dir = ti.Vector(3, ti.f32, ())
        self.camera = ti.Matrix(3, 3, ti.f32, ())
        self.cam_pos = ti.Vector(3, ti.f32, ())
        self.opt = RenderOptions()
        self.balls = []

    def trace(self, pos, dir):
        raise NotImplementedError

    def set_light_dir(self, ldir):
        import math
        norm = math.sqrt(sum(x**2 for x in ldir))
        ldir = [x / norm for x in ldir]
        self.light_dir[None] = ldir

    def set_camera(self, pos=[0, 0, -2], lookat=[0, 0, 0], up=[0, 1, 0]):
        import math
        # fwd = lookat - pos
        fwd = [lookat[i] - pos[i] for i in range(3)]
        # fwd = fwd.normalized()
        fwd_len = math.sqrt(sum(x**2 for x in fwd))
        fwd = [x / fwd_len for x in fwd]
        # right = fwd.cross(up)
        right = [
                fwd[2] * up[1] - fwd[1] * up[2],
                fwd[0] * up[2] - fwd[2] * up[0],
                fwd[1] * up[0] - fwd[0] * up[1],
                ]
        # right = right.normalized()
        right_len = math.sqrt(sum(x**2 for x in right))
        right = [x / right_len for x in right]
        # up = right.cross(fwd)
        up = [
             right[2] * fwd[1] - right[1] * fwd[2],
             right[0] * fwd[2] - right[2] * fwd[0],
             right[1] * fwd[0] - right[0] * fwd[1],
             ]

        # camera = ti.Matrix.cols([right, up, fwd])
        camera = [right, up, fwd]
        camera = [[camera[i][j] for i in range(3)] for j in range(3)]
        self.camera[None] = camera
        self.cam_pos[None] = pos

    @ti.kernel
    def render(self):
        for I in ti.grouped(self.img):
            scale = ti.static(2 / min(*self.img.shape()))
            coor = (I - ts.vec2(*self.img.shape()) / 2) * scale

            orig = ts.vec3(coor, 0.0)
            dir = ts.vec3(0.0, 0.0, 1.0)

            orig = self.camera[None] @ orig + self.cam_pos[None]
            dir = self.camera[None] @ dir

            pos, normal = self.trace(orig, dir)
            light_dir = self.light_dir[None]

            color = self.opt.render_func(pos, normal, dir, light_dir)
            color = self.opt.pre_process(color)
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
