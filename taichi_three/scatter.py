import numpy as np
import taichi as ti
import taichi_glsl as ts
from .geometry import *
from .shading import *
from .transform import *
from .common import *
from .model import *
import math


class ScatterModel(ModelBase):
    def __init__(self, num=None):
        super().__init__()

        self.num = num

        if num is not None:
            self.pos = ti.Vector.field(3, float, num)
            self.radius = ti.field(float, num)

            @ti.materialize_callback
            def initialize_radius():
                self.radius.fill(1.0)

    @ti.func
    def render(self, camera):
        for i in ti.grouped(self.pos):
            render_particle(self, camera, i)

    @ti.func
    def colorize(self, pos, normal):
        with self.material.specify_inputs(model=self, pos=pos, texcoor=None, normal=normal, tangent=None, bitangent=None) as shader:
            return shader.colorize()
