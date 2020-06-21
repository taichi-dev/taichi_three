import taichi as ti
import taichi_glsl as ts
from .geometry import *
import math


@ti.data_oriented
class Model:
    def __init__(self, res=None):
        self.geo_list = []
        self.vertices = []

    @ti.kernel
    def render(self):
        scene = self.scene
        for I in ti.grouped(scene.img):
            scene.img[I] = ts.vec3(0)
        if ti.static(len(self.geo_list)):
            for geo in ti.static(self.geo_list):
                geo.render()

    def add_geometry(self, geom):
        geom.model = self
        self.geo_list.append(geom)

    def set_vertices(self, vert):
        vert.model = self
        self.vertices = vert
