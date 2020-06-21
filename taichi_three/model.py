import taichi as ti
import taichi_glsl as ts
from .geometry import *
from .transform import *
from .common import *
import math


@ti.data_oriented
class Model(AutoInit):
    def __init__(self, obj=None):
        self.geo_list = []
        self.vertices = []
        self.W2L = Affine.var(())
        if obj is not None:
            self.from_obj(obj)

    def _init(self):
        self.W2L.init()

    def from_obj(self, obj):
        vertex = Vertex.var(obj['v'].shape[0])
        face = Face.var(obj['f'].shape[0])
        vertex.pos.from_numpy(obj['v'])
        face.idx.from_numpy(obj['f'])
        self.set_vertices(vertex)
        self.add_geometry(face)

    @ti.func
    def render(self):
        scene = self.scene
        if ti.static(len(self.geo_list)):
            for geo in ti.static(self.geo_list):
                geo.render()

    def add_geometry(self, geom):
        geom.model = self
        self.geo_list.append(geom)

    def set_vertices(self, vert):
        vert.model = self
        self.vertices = vert
