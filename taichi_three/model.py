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
        self.L2W = Affine.var(())
        self.obj = obj
        if obj is not None:
            vertex = Vertex.var(obj['v'].shape[0])
            face = Face.var(obj['f'].shape[0])
            self.set_vertices(vertex)
            self.add_geometry(face)

    def _init(self):
        self.L2W.init()
        if self.obj is not None:
            self.vertices.pos.from_numpy(self.obj['v'])
            self.geo_list[0].idx.from_numpy(self.obj['f'])

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
