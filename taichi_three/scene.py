import taichi as ti
import taichi_glsl as ts
from .transform import *
from .light import *


@ti.data_oriented
class Scene:
    def __init__(self):
        self.lights = []
        self.cameras = []
        self.buffers = []
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
        self.add_camera_d(camera)
        from .buffer import FrameBuffer
        buffer = FrameBuffer(camera)
        self.add_buffer(buffer)

    def add_camera_d(self, camera):
        camera.scene = self
        self.cameras.append(camera)

    def add_buffer(self, buffer):
        buffer.scene = self
        self.buffers.append(buffer)

    def add_light(self, light):
        light.scene = self
        self.lights.append(light)

    @ti.kernel
    def render(self):
        if ti.static(len(self.buffers)):
            for buffer in ti.static(self.buffers):
                buffer.render()
        else:
            ti.static_print('Warning: no cameras / buffers')

    @ti.func
    def pixel_shader(self, mid: ti.template(), pos, tex, nrm, tan, bitan):
        if ti.static(isinstance(mid, int)):
            return ti.static(self.materials[mid]).pixel_shader(pos, tex, nrm, tan, bitan)
        color = ts.vec3(0.0)
        if mid != 0:
            color = ts.vec3(1.0, 0.0, 1.0)  # magenta, for debugging missing materials
        for i, material in ti.static(self.materials.items()):
            if mid == i:
                color = material.pixel_shader(pos, tex, nrm, tan, bitan)
        return color
