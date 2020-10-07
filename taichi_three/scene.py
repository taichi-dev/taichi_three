import taichi as ti
import taichi_glsl as ts
from .transform import *
from .light import *


@ti.data_oriented
class Scene(AutoInit):
    def __init__(self):
        self.lights = []
        self.cameras = []
        self.shadows = []
        self.models = []
        self.curr_camera = None

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

    def add_shadow_camera(self, shadow):
        shadow.scene = self
        self.shadows.append(shadow)

    def add_light(self, light):
        light.scene = self
        self.lights.append(light)

    def _init(self):
        for light in self.lights:
            light.init()
        for camera in self.cameras:
            camera.init()
        for shadow in self.shadows:
            shadow.init()
        for model in self.models:
            model.init()

    def render(self):
        self.init()
        self._render()

    def render_shadows(self):
        self.init()
        self._render_shadows()

    @ti.kernel
    def _render_shadows(self):
        if ti.static(len(self.shadows)):
            for shadow in ti.static(self.shadows):
                self._render_camera(shadow)

    @ti.kernel
    def _render(self):
        if ti.static(len(self.cameras)):
            for camera in ti.static(self.cameras):
                self.curr_camera = ti.static(camera)
                self._render_camera(camera)

        else:
            ti.static_print('Warning: no cameras')
        self.curr_camera = ti.static(None)

    @ti.func
    def _render_camera(self, camera):
        camera.fb.clear_buffer()

        # sets up light directions
        if ti.static(len(self.lights)):
            for light in ti.static(self.lights):
                light.set_view(camera)  # TODO: model.set_view too?
        else:
            ti.static_print('Warning: no lights')

        if ti.static(len(self.models)):
            for model in ti.static(self.models):
                model.render(camera)
        else:
            ti.static_print('Warning: no models')
