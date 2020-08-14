import taichi as ti
import taichi_glsl as ts
from .transform import *
from .shading import *
from .light import *


@ti.data_oriented
class Scene(AutoInit):
    def __init__(self):
        self.lights = []
        self.cameras = []
        self.opt = Shading()
        self.models = []

    def set_light_dir(self, ldir):
        # changes light direction input to the direction
        # from the light towards the object
        # to be consistent with future light types
        if not self.lights:
            light = Light(ldir)
            self.add_light(light)
        else:
            self.light[0].set(ldir)

    @ti.func
    def cook_coor(self, I, camera):
        scale = ti.static(2 / min(*camera.img.shape()))
        coor = (I - ts.vec2(*camera.img.shape()) / 2) * scale
        return coor

    @ti.func
    def uncook_coor(self, coor, camera):
        scale = ti.static(min(*camera.img.shape()) / 2)
        I = coor.xy * scale + ts.vec2(*camera.img.shape()) / 2
        return I

    def add_model(self, model):
        model.scene = self
        self.models.append(model)

    def add_camera(self, camera):
        camera.scene = self
        self.cameras.append(camera)

    def add_light(self, light):
        light.scene = self
        self.lights.append(light)

    def _init(self):
        for light in self.lights:
            light.init()
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
                # sets up light directions
                for light in ti.static(self.lights):
                    light.set_view(camera)
                if ti.static(len(self.models)):
                    for model in ti.static(self.models):
                        model.render(camera)
