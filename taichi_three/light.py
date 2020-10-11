import taichi as ti
import taichi_glsl as ts
from .common import *
from .camera import *
import math
'''
The base light class represents an ambient light.
'''
@ti.data_oriented
class AmbientLight:
    def __init__(self, color=None):
        color = color or [1, 1, 1]
        if not isinstance(color, (list, tuple)):
            color = [color for i in range(3)]
        self.color = ti.Vector.field(3, ti.float32, ())

        @ti.materialize_callback
        def init_light():
            self.color[None] = color

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
class Light:
    shadow = None

    def __init__(self, dir=None, color=None):
        dir = dir or [0, 0, 1]
        norm = math.sqrt(sum(x ** 2 for x in dir))
        dir = [-x / norm for x in dir]
        color = color or [1, 1, 1]
 
        @ti.materialize_callback
        def init_light():
            self.dir[None] = dir
            self.color[None] = color

        self.dir = ti.Vector.field(3, float, ())
        self.color = ti.Vector.field(3, float, ())
        # store the current light direction in the view space
        # so that we don't have to compute it for each vertex
        self.viewdir = ti.Vector.field(3, float, ())

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
        self.viewdir[None] = (camera.L2W[None].inverse() @ ts.vec4(self.dir[None], 0)).xyz

    def make_shadow_camera(self, res=(512, 512), dis=10, fov=60, **kwargs):
        shadow = Camera(res=res)
        shadow.fov = math.radians(fov)
        #shadow.ctl = CameraCtl(pos=(-self.dir[None].value * dis).entries, **kwargs)
        #@ti.materialize_callback
        def init_camera():
            shadow.ctl.apply(shadow)
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

        lspos = (self.shadow.L2W[None] @ ts.vec4(wpos, 1)).xyz
        lscoor = self.shadow.uncook(lspos)

        cur_idepth = self.shadow.fb.idepth_fixp(1 / lspos.z)

        l = self._sub_SDlerp(cur_idepth, lscoor, ts.D.X_)
        r = self._sub_SDlerp(cur_idepth, lscoor, ts.D.x_)
        t = self._sub_SDlerp(cur_idepth, lscoor, ts.D._x)
        b = self._sub_SDlerp(cur_idepth, lscoor, ts.D._X)
        c = self._sub_SDlerp(cur_idepth, lscoor, ts.D.__)
        return (l + r + t + b + c * 4) / 8



class PointLight(Light):
    def __init__(self, pos=None, color=None, c1=None, c2=None):
        pos = pos or [0, 0, -2]
        color = color or [1, 1, 1]
        self.c1 = c1 or 1
        self.c2 = c2 or 1
        self.pos = ti.Vector.field(3, float, ())
        self.color = ti.Vector.field(3, float, ())
        self.viewpos = ti.Vector.field(3, float, ())

        @ti.materialize_callback
        def init_light():
            self.pos[None] = pos
            self.color[None] = color

    @ti.func
    def set_view(self, camera):
        # TODO: merge t3.PointLight with t3.Light by considering `w`?
        self.viewpos[None] = (camera.L2W[None].inverse() @ ts.vec4(self.pos[None], 1)).xyz

    @ti.func
    def intensity(self, pos):
        distsq = (self.viewpos[None] - pos).norm_sqr()
        return 1 / (1 + self.c1 * ti.sqrt(distsq) + self.c2 * distsq)

    @ti.func
    def get_dir(self, pos):
        return ts.normalize(self.viewpos[None] - pos)
