import taichi as ti
import taichi_glsl as ts
from .geometry import *
import math


@ti.data_oriented
class Model:
    def __init__(self, obj=None):
        self.geo_list = []
        self.vertices = []
        if obj is not None:
            self.from_obj(obj)

    def from_obj(self, obj):
        import taichi_three as t3
        vertex = t3.Vertex.var(obj['v'].shape[0])
        face = t3.Face.var(obj['f'].shape[0])
        vertex.pos.from_numpy(obj['v'])
        face.idx.from_numpy(obj['f'])
        self.set_vertices(vertex)
        self.add_geometry(face)

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
