import taichi as ti
import taichi_glsl as ts
from .common import *
from .camera import *
import math

'''
The base light class represents a directional light.
'''
@ti.data_oriented
class Light(AutoInit):

    def __init__(self, dir=None, color=None):
        dir = dir or [0, 0, 1]
        norm = math.sqrt(sum(x ** 2 for x in dir))
        dir = [x / norm for x in dir]
 
        self.dir_py = [-x for x in dir]
        self.color_py = color or [1, 1, 1] 

        self.dir = ti.Vector.field(3, ti.float32, ())
        self.color = ti.Vector.field(3, ti.float32, ())
        # store the current light direction in the view space
        # so that we don't have to compute it for each vertex
        self.viewdir = ti.Vector.field(3, ti.float32, ())

        self.shadow = None

    def set(self, dir=[0, 0, 1], color=[1, 1, 1]):
        norm = math.sqrt(sum(x**2 for x in dir))
        dir = [x / norm for x in dir]
        self.dir_py = dir
        self.color = color

    def _init(self):
        self.dir[None] = self.dir_py
        self.color[None] = self.color_py

    @ti.func
    def intensity(self, pos):
        return 1

    @ti.func
    def get_color(self, pos):
        return self.color[None] * self.intensity(pos)

    @ti.func
    def get_dir(self, pos):
        return self.viewdir[None]

    @ti.func
    def set_view(self, camera):
        self.viewdir[None] = camera.untrans_dir(self.dir[None])

    def make_shadow_camera(self, dis=10, fov=60, **kwargs):
        shadow = Camera(pos=[x * dis for x in self.dir_py], fov=fov, **kwargs)
        shadow.type = shadow.ORTHO
        self.shadow = shadow
        return shadow

    @ti.func
    def _sub_SO(self, cur_idepth, lscoor):
        lst_idepth = ts.sample(self.shadow.fb['idepth'], lscoor)
        return 1 if lst_idepth < cur_idepth + 1e-4 else 0

    @ti.func
    def shadow_occlusion(self, wpos):
        if ti.static(self.shadow is None):
            return 1

        lspos = self.shadow.untrans_pos(wpos)
        lscoor = self.shadow.uncook(lspos)

        cur_idepth = 1 / lspos.z
        x = ts.fract(lscoor)
        y = 1 - x
        B = int(lscoor)
        xx = self._sub_SO(cur_idepth, B + ts.D.xx)
        x_ = self._sub_SO(cur_idepth, B + ts.D.x_)
        __ = self._sub_SO(cur_idepth, B + ts.D.__)
        _x = self._sub_SO(cur_idepth, B + ts.D._x)
        return xx * x.x * x.y + x_ * x.x * y.y + __ * y.x * y.y + _x * y.x * x.y



class PointLight(Light):
    def __init__(self, position=None, color=None,
            c1=None, c2=None):
        position = position or [0, 1, -3]
        if c1 is not None:
            self.c1 = c1
        if c2 is not None:
            self.c2 = c2
        self.pos_py = position
        self.color_py = color or [1, 1, 1]
        self.pos = ti.Vector.field(3, ti.f32, ())
        self.color = ti.Vector.field(3, ti.f32, ())
        self.viewpos = ti.Vector.field(3, ti.f32, ())

    def _init(self):
        self.pos[None] = self.pos_py
        self.color[None] = self.color_py

    @ti.func
    def set_view(self, camera):
        self.viewpos[None] = camera.untrans_pos(self.pos[None])

    @ti.func
    def intensity(self, pos):
        distsq = (self.viewpos[None] - pos).norm_sqr()
        return 1. / (1. + self.c1 * ti.sqrt(distsq) + self.c2 * distsq)

    @ti.func
    def get_dir(self, pos):
        return ts.normalize(self.viewpos[None] - pos)
