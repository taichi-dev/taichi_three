import taichi as ti
import taichi_glsl as ts
from .common import *
import math

@ti.data_oriented
class Light(AutoInit):

    def __init__(self, dir=None, color=None):
        dir = dir or [0, 0, 1]
        norm = math.sqrt(sum(x ** 2 for x in dir))
        dir = [x / norm for x in dir]

        self.np_dir = [-x for x in dir]
        self.np_color = color or [1, 1, 1]

        self.dir = ti.Vector(3, ti.float32, ())
        self.color = ti.Vector(3, ti.float32, ())

    def set(self, dir=[0, 0, 1], color=[1, 1, 1]):
        norm = math.sqrt(sum(x**2 for x in dir))
        dir = [x / norm for x in dir]
        self.np_dir = dir
        self.color = color

    def _init(self):

        
        self.dir[None] = self.np_dir
        self.color[None] = self.np_color