import taichi as ti
import taichi_glsl as ts
from .objects import Triangle
from .scene import *
import math


@ti.data_oriented
class SceneGE(SceneBase):
    def __init__(self, res=None):
        super(SceneGE, self).__init__(res)
        self.triangles = []

    @ti.kernel
    def do_render(self):
        for tri in ti.static(self.triangles):
            tri.render(self)

    def add_triangle(self, a, b, c):
        tri = Triangle(a, b, c)
        self.triangles.append(tri)
