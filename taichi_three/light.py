import taichi as ti
import taichi_glsl as ts
from .common import *
import math

'''
The base light class represents a directional light.
'''
@ti.data_oriented
class Light(AutoInit):

    def __init__(self, direction=None, color=None):
        direction = direction or [0, 0, 1]
        norm = math.sqrt(sum(x ** 2 for x in direction))
        direction = [x / norm for x in direction]
 
        self.dir_py = [-x for x in direction]
        self.color_py = color or [1, 1, 1] 

        self.dir = ti.Vector(3, ti.float32, ())
        self.color = ti.Vector(3, ti.float32, ())
        # store the current light direction in the view space
        # so that we don't have to compute it for each vertex
        self.viewdir = ti.Vector(3, ti.float32, ())

    def set(self, direction=[0, 0, 1], color=[1, 1, 1]):
        norm = math.sqrt(sum(x**2 for x in direction))
        direction = [x / norm for x in direction]
        self.dir_py = direction
        self.color = color

    def _init(self):
        self.dir[None] = self.dir_py
        self.color[None] = self.color_py

    @ti.func
    def intensity(self, pos):
        return 1

    @ti.func
    def get_color(self, pos):
        return self.color[None] * self.intensity(pos)

    @ti.func
    def get_dir(self, pos):
        return self.viewdir

    @ti.func
    def set_view(self, camera):
        self.viewdir[None] = camera.untrans_dir(self.dir[None])



class PointLight(Light):

    def __init__(self, position=None, color=None,
            c1=None, c2=None):
        position = position or [0, 1, -3]
        if c1 is not None: 
            self.c1 = c1
        if c2 is not None: 
            self.c2 = c2
        self.pos_py = position
        self.color_py = color or [1, 1, 1] 
        self.pos = ti.Vector(3, ti.float32, ())
        self.color = ti.Vector(3, ti.float32, ())
        self.viewpos = ti.Vector(3, ti.float32, ())

    def _init(self):
        self.pos[None] = self.pos_py
        self.color[None] = self.color_py

    @ti.func
    def set_view(self, camera):
        self.viewpos[None] = camera.untrans_pos(self.pos[None])

    @ti.func
    def intensity(self, pos):
        distsq = (self.viewpos[None] - pos).norm_sqr()
        return 1. / (1. + self.c1 * ti.sqrt(distsq) + self.c2 * distsq)

    @ti.func
    def get_dir(self, pos):
        return ts.normalize(self.viewpos[None] - pos)
    
    

