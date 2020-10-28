import taichi as ti
import taichi_glsl as ts
from .common import *
from .camera import *
from .transform import *
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

    def shadow_occlusion(self, wpos, normal):
        return 1


'''
The base light class represents a directional light.
'''
@ti.data_oriented
class Light:
    shadow = None

    def __init__(self, dir=None, color=None):
        dir = dir or [0, 0, -1]
        norm = math.sqrt(sum(x ** 2 for x in dir))
        dir = [-x / norm for x in dir]
        color = color or [1, 1, 1]
        if not isinstance(color, (list, tuple)):
            color = [color for i in range(3)]
 
        @ti.materialize_callback
        @ti.kernel
        def init_light():
            self.dir[None] = dir
            self.color[None] = color
            #self.L2W[None] = transform(makeortho(self.dir[None]), self.dir[None] * 2)

        self.dir = ti.Vector.field(3, float, ())
        self.color = ti.Vector.field(3, float, ())
        #self.L2W = ti.Matrix.field(4, 4, float, ())
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
        self.viewdir[None] = v4trans(camera.L2W[None].inverse(), self.dir[None], 0)

    def make_shadow_camera(self, res=(512, 512), fov=None, distance=2, **kwargs):
        shadow = Camera(res=res, fov=fov or math.radians(60))
        from .buffer import FrameBuffer
        fb = FrameBuffer(shadow, buffers=dict())  # only record depth info
        shadow.ctl = None
        shadow.type = shadow.ORTHO
        shadow.distance = distance
        ti.materialize_callback(self.update_shadow)
        self.shadow = shadow
        return shadow

    @ti.kernel
    def update_shadow(self):
        self.shadow.L2W[None] = transform(makeortho(self.dir[None]), self.dir[None] * self.shadow.distance) @ scale(-1, 1, 1)

    @ti.func
    def _sub_SO(self, cur_z, lscoor):
        lst_z = 1 / (1e-6 + ts.sample(self.shadow.fb['idepth'], lscoor))
        return 1 if lst_z > cur_z - 1e-3 else 0

    @ti.func
    def _sub_SDlerp(self, cur_z, lscoor, D):
        x = ts.fract(lscoor)
        y = 1 - x
        B = int(lscoor)
        xx = self._sub_SO(cur_z, B + D + ts.D.xx)
        xy = self._sub_SO(cur_z, B + D + ts.D.xy)
        yy = self._sub_SO(cur_z, B + D + ts.D.yy)
        yx = self._sub_SO(cur_z, B + D + ts.D.yx)
        return max(xx, xy, yy, yx)
        #return xx * x.x * x.y + xy * x.x * y.y + yy * y.x * y.y + yx * y.x * x.y

    @ti.func
    def shadow_occlusion(self, wpos, normal):
        if ti.static(self.shadow is None):
            return 1

        lspos = v4trans(self.shadow.L2W[None].inverse(), wpos, 1)
        lscoor = self.shadow.uncook(lspos)

        cur_z = lspos.z  # TODO: bend with normal

        return self._sub_SDlerp(cur_z, lscoor, ts.vec2(0))

        '''
        res = 0.0
        for d1, d2 in ti.ndrange(2, 2):
            d = d1 * 2 - 1
            D = ts.vec2(d, 0) if d2 == 0 else ts.vec2(0, d)
            res += self._sub_SDlerp(cur_z, lscoor, D)
        res += 4 * self._sub_SDlerp(cur_z, lscoor, ts.vec2(0))
        return res / 8
        '''



class PointLight(Light):
    def __init__(self, pos=None, color=None, c1=None, c2=None):
        pos = pos or [0, 0, -2]
        color = color or [1, 1, 1]
        if not isinstance(color, (list, tuple)):
            color = [color for i in range(3)]
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
