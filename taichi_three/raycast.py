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
        self.rc = ti.Vector.field(3, float, self.N)
        self.rI = ti.Vector.field(2, int, self.N)

    @ti.kernel
    def steprays(self):
        for i in self.ro:
            hit = 1e6
            orig = self.ro[i]
            dir = self.rd[i]
            clr = ts.vec3(0.0)
            for model in ti.static(self.scene.models):
                ihit, iorig, idir, iclr = model.intersect(self.ro[i], self.rd[i])
                if ihit < hit:
                    hit, orig, dir, clr = ihit, iorig + idir * 1e-4, idir, iclr
            self.ro[i] = orig
            self.rd[i] = dir
            self.rc[i] *= clr

    @ti.kernel
    def loadrays(self):
        self.fb.clear_buffer()
        for I in ti.grouped(ti.ndrange(*self.res)):
            i = I.dot(ts.vec(1, self.res[0]))
            coor = ts.vec2((I.x - self.cx) / self.fx, (I.y - self.cy) / self.fy)
            orig, dir = self.generate(coor)
            self.ro[i] = orig
            self.rd[i] = dir
            self.rc[i] = ts.vec3(1.0)
            self.rI[i] = I

    @ti.kernel
    def applyrays(self):
        for i in self.ro:
            self.img[self.rI[i]] = self.rc[i]

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