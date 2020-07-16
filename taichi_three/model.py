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
            vertex_tex = VertexTex.var(obj['vt'].shape[0])
            self.set_vertices(vertex)
            self.set_textures(vertex_tex)
            self.add_geometry(face)
            self.texture = ti.Vector(3, ti.f32, (obj['texture'].shape[0], obj['texture'].shape[1]))

    def _init(self):
        self.L2W.init()
        if self.obj is not None:
            self.vertices.pos.from_numpy(self.obj['v'])
            self.geo_list[0].idx.from_numpy(self.obj['f'])
            self.vertex_tex.pos.from_numpy(self.obj['vt'])
            self.texture.from_numpy(self.obj['texture'])

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

    def set_textures(self, vertex_tex):
        vertex_tex.model = self
        self.vertex_tex = vertex_tex
