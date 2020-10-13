import taichi as ti
import taichi_glsl as ts
from .common import *
from .geometry import *
from .camera import *


@ti.data_oriented
class Accumator:
    def __init__(self, shape=(512, 512)):
        self.buf = ti.Vector.field(3, float, shape)
        self.count = ti.field(int, ())

    @ti.kernel
    def accumate(self, src: ti.template()):
        self.count[None] += 1
        alpha = max(1 / 512, 1 / self.count[None])
        for I in ti.grouped(self.buf):
            self.buf[I] = self.buf[I] * (1 - alpha) + src[I] * alpha

    @ti.kernel
    def reset(self):
        self.count[None] = 0
        for I in ti.grouped(self.buf):
            self.buf[I] *= 0

    @ti.kernel
    def denoise(self, alpha: ti.template()):
        ti.static_print('denoise', alpha)
        if ti.static(alpha != 0):
            for I in ti.grouped(self.buf):
                center = ts.clamp(self.buf[I])
                around = ts.clamp((self.buf[I + ts.D.x_] + self.buf[I + ts.D.X_] + self.buf[I + ts.D._x] + self.buf[I + ts.D._X]) / 4)
                #amax = ts.clamp(max(self.buf[I + ts.D.x_], self.buf[I + ts.D.X_], self.buf[I + ts.D._x], self.buf[I + ts.D._X]))
                #amin = ts.clamp(min(self.buf[I + ts.D.x_], self.buf[I + ts.D.X_], self.buf[I + ts.D._x], self.buf[I + ts.D._X]))
                #if center <= amin + throttle or center >= amax - throttle:
                self.buf[I] = center * (1 - alpha) + around * alpha

    def render(self, camera, depth, baseres=2, regrate=32):
        rate = max(0, baseres - self.count[None] // regrate)
        region = camera.res[0] // 2**rate, camera.res[1] // 2**rate
        camera.loadrays((0, 0), region, 2**rate)
        for step in range(depth):
            camera.steprays()
        camera.applyrays()
        self.accumate(camera.img)
        #self.denoise(0.1 * (rate / baseres))


class RTCamera(Camera):
    def __init__(self, res=None):
        super().__init__(res)

        self.N = self.res[0] * self.res[1]
        self.ro = ti.Vector.field(3, float, self.N)
        self.rd = ti.Vector.field(3, float, self.N)
        self.rc = ti.Vector.field(3, float, self.N)
        self.rI = ti.Vector.field(2, int, self.N)

    def steprays(self):
        nrays = self.region[0] * self.region[1]
        self._steprays(nrays)

    @ti.kernel
    def _steprays(self, nrays: ti.template()):
        for i in range(nrays):
            hit = 1e6
            orig, dir = self.ro[i], self.rd[i]
            if self.rd[i].norm_sqr() >= 1e-3:
                clr = ts.vec3(0.0)
                for model in ti.static(self.scene.models):
                    ihit, iorig, idir, iclr = model.intersect(self.ro[i], self.rd[i])
                    if ihit < hit:
                        hit, orig, dir, clr = ihit, iorig + idir * 1e-4, idir, iclr
                self.ro[i], self.rd[i] = orig, dir
                self.rc[i] *= clr

    def loadrays(self, topleft=None, region=None, skipstep=None):
        self.topleft = topleft or (0, 0)
        self.region = region or self.res
        self.skipstep = skipstep or 1
        self._loadrays(self.topleft, self.region, self.skipstep)

    @ti.kernel
    def _loadrays(self, topleft: ti.template(), region: ti.template(), skipstep: ti.template()):
        ti.static_print('loadrays:', topleft, region, skipstep)
        for II in ti.grouped(ti.ndrange(*region)):
            I = II * skipstep + topleft
            for J in ti.static(ti.grouped(ti.ndrange(skipstep, skipstep))):
                self.img[I + J] *= 0
        for II in ti.grouped(ti.ndrange(*region)):
            i = II.dot(ts.vec(1, region[0]))
            I = II * skipstep + topleft + skipstep / 2
            coor = ts.vec2((I.x - self.cx) / self.fx, (I.y - self.cy) / self.fy)
            orig, dir = self.generate(coor)
            self.ro[i] = orig
            self.rd[i] = dir
            self.rc[i] = ts.vec3(1.0)
            self.rI[i] = II

    def applyrays(self):
        nrays = self.region[0] * self.region[1]
        self._applyrays(nrays, self.topleft, self.skipstep)

    @ti.kernel
    def _applyrays(self, nrays: ti.template(), topleft: ti.template(), skipstep: ti.template()):
        for i in range(nrays):
            I = self.rI[i] * skipstep + topleft
            for J in ti.static(ti.grouped(ti.ndrange(skipstep, skipstep))):
                self.img[I + J] = self.rc[i]

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