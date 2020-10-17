import taichi as ti
import taichi_glsl as ts
from .common import *
from .transform import *
import math



@ti.data_oriented
class FrameBuffer:
    def __init__(self, camera, dim=3, taa=False):
        self.camera = camera
        camera.fb = self
        self.res = camera.res
        self.n_taa = (taa if not isinstance(taa, bool) else 5) if taa else 0
        if self.n_taa:
            assert self.n_taa >= 2
            self.taa = ti.Vector.field(3, float, (self.n_taa, *res))
            self.itaa = ti.field(int, ())
            self.ntaa = ti.field(int, ())

        self.buffers = {}
        self.add_buffer('img', dim)
        self.add_buffer('idepth', ())

    @ti.func
    def render(self):
        self.clear_buffer()
        self.camera.render()
        self.update_buffer()

    @ti.func
    def idepth_fixp(self, z):
        if ti.static(ti.core.is_integral(self['idepth'].dtype)):
            return int(2**24 * z)
        else:
            return z

    @ti.func
    def atomic_depth(self, X, depth):
        idepth = self.idepth_fixp(1 / depth)
        return idepth < ti.atomic_max(self['idepth'][X], idepth)

    def add_buffer(self, name, dim, dtype=float):
        buf = create_field(dim, dtype, self.res)
        self.buffers[name] = buf

    @subscriptable
    @ti.func
    def _taaimg(self, i, j):
        return self.taa[self.itaa[None], i, j]

    @property
    def img(self):
        return self.buffers['img']

    @property
    def idepth(self):
        return self.buffers['idepth']

    def __getitem__(self, name):
        if isinstance(name, tuple):
            name = name[0]
        if self.n_taa and name == 'img':
            return self._taaimg
        if name in self.buffers:
            return self.buffers[name]
        else:
            return dummy_expression()

    @ti.func
    def update(self, I, data: ti.template()):
        for k, v in ti.static(data.items()):
            self[k][I] = v

    def fetchpixelinfo(self, name, pos):
        if name in self.buffers:
            I = int(pos[0] * self.res[0]), int(pos[1] * self.res[1])
            return self[name][I].value
        else:
            return None

    @ti.func
    def clear_buffer(self):
        if ti.static(self.n_taa):
            self.ntaa[None] = min(self.n_taa, self.ntaa[None] + 1)
            self.itaa[None] = (self.itaa[None] + 1) % self.n_taa
        for I in ti.grouped(self.img):
            for k in ti.static(self.buffers.keys()):
                self[k][I] *= 0

    @ti.kernel
    def flush_taa(self):
        if ti.static(self.n_taa):
            self.ntaa[None] = 0
            self.itaa[None] = 0
            for I in ti.grouped(self.img):
                r = ts.vec3(0.0)
                for i in ti.static(range(self.n_taa)):
                    self.taa[i, I] *= 0

    @ti.func
    def update_buffer(self):
        if ti.static(self.n_taa):
            for I in ti.grouped(self.img):
                self.img[I] = sum(self.taa[i, I] for i in range(self.n_taa)) / self.ntaa[None]


@ti.data_oriented
class DeferredShading:
    def __init__(self, src, material, dim=3):
        self.res = src.res
        self.img = create_field(dim, float, self.res)
        self.material = material
        self.src = src

    @ti.func
    def render(self):
        self.src.render()
        for i in ti.grouped(self.img):
            pos = ts.vec3(0.0)
            texcoor = ts.vec2(0.0)
            normal = ts.vec3(0.0)
            tangent = ts.vec3(0.0)
            bitangent = ts.vec3(0.0)
            unpack_tuple(self.src.img[i], pos, texcoor, normal, tangent, bitangent)
            color = self.material.pixel_shader(self, pos, texcoor, normal, tangent, bitangent)
            self.img[i] = color


@ti.data_oriented
class SuperSampling2x2:
    def __init__(self, src, dim=3):
        self.res = (src.res[0] // 2, src.res[1] // 2)
        self.img = create_field(dim, float, self.res)
        self.src = src

    @ti.func
    def render(self):
        self.src.render()
        for i in ti.grouped(self.img):
            self.img[i] = (self.src.img[i * 2 + ts.D.__]
                         + self.src.img[i * 2 + ts.D._x]
                         + self.src.img[i * 2 + ts.D.x_]
                         + self.src.img[i * 2 + ts.D.xx]) / 4


# Used for camera.fb.post_process
# https://zhuanlan.zhihu.com/p/21983679
def make_tonemap(adapted_lum=1.2):
    @ti.func
    def result(color):
        color *= adapted_lum
        return color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14)