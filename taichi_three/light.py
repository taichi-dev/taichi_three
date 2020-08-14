import taichi as ti
import taichi_glsl as ts
from .common import *
import math


@ti.data_oriented
class Light(AutoInit):

    def __init__(self, dir=[0, 0, 1], color=[1, 1, 1]):
        norm = math.sqrt(sum(x ** 2 for x in dir))
        dir = [x / norm for x in dir]

        self.dir_py = [-x for x in dir]
        self.color_py = color

        self.dir = ti.Vector(3, ti.float32, ())
        self.color = ti.Vector(3, ti.float32, ())
        self.set(init=True)

    def set(self, dir=None, color=None, init=False):
        dir = dir or self.dir_py
        color = color or self.color_py
        norm = math.sqrt(sum(x**2 for x in dir))
        dir = [x / norm for x in dir]
        self.dir_py = dir
        self.color_py = color
        if not init:
            self._init()

    def _init(self):
        self.dir[None] = self.dir_py
        self.color[None] = self.color_py
