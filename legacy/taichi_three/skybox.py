import numpy as np
import taichi as ti
import taichi_glsl as ts
from .shading import *
from .model import *
import math


class Skybox(ModelBase):
    def __init__(self, texture, scale=None):
        super().__init__()

        if isinstance(texture, str):
            texture = ti.imread(texture)

        # convert UInt8 into Float32 for storage:
        if texture.dtype == np.uint8:
            texture = texture.astype(np.float32) / 255
        elif texture.dtype == np.float64:
            texture = texture.astype(np.float32)

        if len(texture.shape) == 3 and texture.shape[2] == 1:
            texture = texture.reshape(texture.shape[:2])

        # either RGB or greyscale
        if len(texture.shape) == 2:
            self.texture = ti.field(float, texture.shape)

        else:
            assert len(texture.shape) == 3, texture.shape
            texture = texture[:, :, :3]
            assert texture.shape[2] == 3, texture.shape

            if scale is not None:
                if callable(scale):
                    texture = scale(texture)
                else:
                    texture *= np.array(scale)[None, None, ...]

            # TODO: use create_field for this
            self.texture = ti.Vector.field(3, float, texture.shape[:2])

        @ti.materialize_callback
        def init_texture():
            self.texture.from_numpy(texture)

    @ti.func
    def render(self, camera):
        for I in ti.grouped(ti.ndrange(*camera.res)):
            if camera.fb.idepth[I] != 0:
                continue
            id = I / ts.vec(*camera.res) * 2 - 1
            dir = ts.vec3(id * ti.tan(camera.fov), 1.0)
            dir = v4trans(self.L2C[None].inverse(), dir, 0).normalized()
            color = self.sample(dir)
            camera.fb.update(I, dict(img=color))

    @ti.func
    def sample(self, dir):
        I = ts.vec2(0.0)
        eps = 1e-5
        dps = 1 - 12 / self.texture.shape[0]
        if dir.z >= 0 and dir.z >= abs(dir.y) - eps and dir.z >= abs(dir.x) - eps:
            I = ts.vec(3 / 8, 3 / 8) + dir.xy / dir.z / 8 * dps
        if dir.z <= 0 and -dir.z >= abs(dir.y) - eps and -dir.z >= abs(dir.x) - eps:
            I = ts.vec(7 / 8, 3 / 8) + dir.Xy / -dir.z / 8 * dps
        if dir.x <= 0 and -dir.x >= abs(dir.y) - eps and -dir.x >= abs(dir.z) - eps:
            I = ts.vec(1 / 8, 3 / 8) + dir.zy / -dir.x / 8 * dps
        if dir.x >= 0 and dir.x >= abs(dir.y) - eps and dir.x >= abs(dir.z) - eps:
            I = ts.vec(5 / 8, 3 / 8) + dir.Zy / dir.x / 8 * dps
        if dir.y >= 0 and dir.y >= abs(dir.x) - eps and dir.y >= abs(dir.z) - eps:
            I = ts.vec(3 / 8, 5 / 8) + dir.xZ / dir.y / 8 * dps
        if dir.y <= 0 and -dir.y >= abs(dir.x) - eps and -dir.y >= abs(dir.z) - eps:
            I = ts.vec(3 / 8, 1 / 8) + dir.xz / -dir.y / 8 * dps
        I = ts.vec2(self.texture.shape[0]) * I
        return ts.bilerp(self.texture, I)