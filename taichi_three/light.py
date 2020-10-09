import taichi as ti
import taichi_glsl as ts
from .common import *
from .camera import *
import math
'''
The base light class represents an ambient light.
'''
@ti.data_oriented
class AmbientLight(AutoInit):
    def __init__(self, color=None):
        self.color_py = color or [1, 1, 1]
        if not isinstance(self.color_py, (list, tuple)):
            self.color_py = [self.color_py for i in range(3)]
        self.color = ti.Vector.field(3, ti.float32, ())

    def _init(self):
        self.color[None] = self.color_py

    def set_view(self, camera):
        pass

    @ti.func
    def get_color(self, pos):
        return self.color[None]

    def shadow_occlusion(self, wpos):
        return 1


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
        self.viewdir[None] = (camera.L2W[None].inverse() @ ts.vec4(self.dir[None], 0)).xyz  # TODO: merge t3.PointLight with t3.Light by considering `w`?

    def make_shadow_camera(self, dis=10, fov=60, **kwargs):
        shadow = Camera(pos=[x * dis for x in self.dir_py], fov=fov, **kwargs)
        shadow.type = shadow.ORTHO
        self.shadow = shadow
        return shadow

    @ti.func
    def _sub_SO(self, cur_idepth, lscoor):
        lst_idepth = ts.sample(self.shadow.fb['idepth'], lscoor)
        return 1 if lst_idepth < cur_idepth + self.shadow.fb.idepth_fixp(1e-3) else 0

    @ti.func
    def _sub_SDlerp(self, cur_idepth, lscoor, D):
        x = ts.fract(lscoor)
        y = 1 - x
        B = int(lscoor)
        xx = self._sub_SO(cur_idepth, B + D + ts.D.xx)
        xy = self._sub_SO(cur_idepth, B + D + ts.D.xy)
        yy = self._sub_SO(cur_idepth, B + D + ts.D.yy)
        yx = self._sub_SO(cur_idepth, B + D + ts.D.yx)
        return xx * x.x * x.y + xy * x.x * y.y + yy * y.x * y.y + yx * y.x * x.y

    @ti.func
    def shadow_occlusion(self, wpos):
        if ti.static(self.shadow is None):
            return 1

        lspos = self.shadow.untrans_pos(wpos)
        lscoor = self.shadow.uncook(lspos)

        cur_idepth = self.shadow.fb.idepth_fixp(1 / lspos.z)

        l = self._sub_SDlerp(cur_idepth, lscoor, ts.D.X_)
        r = self._sub_SDlerp(cur_idepth, lscoor, ts.D.x_)
        t = self._sub_SDlerp(cur_idepth, lscoor, ts.D._x)
        b = self._sub_SDlerp(cur_idepth, lscoor, ts.D._X)
        c = self._sub_SDlerp(cur_idepth, lscoor, ts.D.__)
        return (l + r + t + b + c * 4) / 8



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
        self.pos = ti.Vector.field(3, float, ())
        self.color = ti.Vector.field(3, float, ())
        self.viewpos = ti.Vector.field(3, float, ())

    def _init(self):
        self.pos[None] = self.pos_py
        self.color[None] = self.color_py

    @ti.func
    def set_view(self, camera):
        self.viewdir[None] = (camera.L2W[None].inverse() @ ts.vec4(self.pos[None], 1)).xyz  # TODO: merge t3.PointLight with t3.Light by considering `w`?

    @ti.func
    def intensity(self, pos):
        distsq = (self.viewpos[None] - pos).norm_sqr()
        return 1 / (1 + self.c1 * ti.sqrt(distsq) + self.c2 * distsq)

    @ti.func
    def get_dir(self, pos):
        return ts.normalize(self.viewpos[None] - pos)
