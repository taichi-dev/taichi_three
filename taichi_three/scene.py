import taichi as ti
import taichi_glsl as ts
from .render import *


@ti.data_oriented
class Scene:
    def __init__(self, res=None):
        self.res = res or (512, 512)
        self.img = ti.Vector.var(3, ti.f32, self.res)
        self.zbuf = ti.var(ti.f32, self.res)
        self.light_dir = ti.Vector.var(3, ti.f32, ())
        self.camera = Camera()
        self.opt = Shader()
        self.models = []

    def set_light_dir(self, ldir):
        norm = math.sqrt(sum(x**2 for x in ldir))
        ldir = [x / norm for x in ldir]
        self.light_dir[None] = ldir

    @ti.func
    def cook_coor(self, I):
        scale = ti.static(2 / min(*self.img.shape()))
        coor = (I - ts.vec2(*self.img.shape()) / 2) * scale
        return coor

    @ti.func
    def uncook_coor(self, coor):
        scale = ti.static(min(*self.img.shape()) / 2)
        I = coor.xy * scale + ts.vec2(*self.img.shape()) / 2
        return I

    def add_model(self, model):
        model.scene = self
        self.models.append(model)

    def render(self):
        if not self.camera.is_set:
            self.camera.set()
        self._render()

    @ti.kernel
    def _render(self):
        for I in ti.grouped(self.img):
            self.img[I] = ts.vec3(0.0)
            self.zbuf[I] = 0.0
        if ti.static(len(self.models)):
            for model in ti.static(self.models):
                model.render()
