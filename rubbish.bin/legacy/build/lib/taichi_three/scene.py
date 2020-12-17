import taichi as ti
import taichi_glsl as ts
from .transform import *
from .light import *


@ti.data_oriented
class Scene(AutoInit):
    def __init__(self):
        self.lights = []
        self.cameras = []
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
                camera.fb.clear_buffer()

                # sets up light directions
                if ti.static(len(self.lights)):
                    for light in ti.static(self.lights):
                        light.set_view(camera)
                else:
                    ti.static_print('Warning: no lights')

                if ti.static(len(self.models)):
                    for model in ti.static(self.models):
                        model.render(camera)
                else:
                    ti.static_print('Warning: no models')

        else:
            ti.static_print('Warning: no cameras')
