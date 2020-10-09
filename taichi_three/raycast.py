import taichi as ti
import taichi_glsl as ts
from .common import *
from .geometry import *
from .camera import *


class RTCamera(Camera):
    def __init__(self, res=None):
        super().__init__(res)

        self.N = self.res[0] * self.res[1]
        self.ro = ti.Vector.field(3, float, self.N)
        self.rd = ti.Vector.field(3, float, self.N)
        self.rI = ti.Vector.field(2, int, self.N)

    @ti.kernel
    def accumate(self):
        self.scene.curr_camera = ti.static(self)
        self.fb.clear_buffer()
        self.scene.curr_camera = ti.static(None)

    @ti.kernel
    def steprays(self):
        for model in ti.static(self.scene.models):
            for i in self.ro:
                model.intersect(self, self.rI[i], self.ro[i], self.rd[i])

    @ti.kernel
    def loadrays(self):
        self.fb.clear_buffer()
        for I in ti.grouped(ti.ndrange(*self.res)):
            i = I.dot(ts.vec(1, self.res[0]))
            coor = ts.vec2((I.x - self.cx) / self.fx, (I.y - self.cy) / self.fy)
            orig, dir = self.generate(coor)
            #print(i, orig, dir)
            self.ro[i] = orig
            self.rd[i] = dir
            self.rI[i] = I

    @ti.func
    def generate(self, coor):
        orig = ts.vec3(0.0)
        dir  = ts.vec3(0.0, 0.0, 1.0)

        if ti.static(self.type == self.ORTHO):
            orig = ts.vec3(coor, 0.0)
        elif ti.static(self.type == self.TAN_FOV):
            uv = coor * self.fov
            dir = ts.normalize(ts.vec3(uv, 1))
        elif ti.static(self.type == self.COS_FOV):
            uv = coor * self.fov
            dir = ts.vec3(ti.sin(uv), ti.cos(uv.norm()))

        orig = (self.L2W[None] @ ts.vec4(orig, 1)).xyz
        dir = (self.L2W[None] @ ts.vec4(dir, 0)).xyz

        return orig, dir