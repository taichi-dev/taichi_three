import taichi as ti
import taichi_glsl as ts
from .transform import *
from .shading import *


@ti.data_oriented
class Scene(AutoInit):
    def __init__(self):
        self.light_dir = ti.Vector.var(3, ti.f32, ())
        self.cameras = []
        self.opt = Shading()
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

    def add_camera(self, camera):
        camera.scene = self
        self.cameras.append(camera)

    def _init(self):
        for camera in self.cameras:
            camera.init()
        for model in self.models:
            model.init()

    def render(self):
        self.init()
        self._render()

    @ti.kernel
    def _render(self):
        if ti.static(len(self.cameras)):
            for camera in ti.static(self.cameras):
                camera.clear_buffer()
                if ti.static(len(self.models)):
                    for model in ti.static(self.models):
                        model.render(camera)
