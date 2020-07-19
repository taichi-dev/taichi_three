import numpy as np
import taichi as ti
import taichi_glsl as ts
from .geometry import *
from .transform import *
from .common import *
import math


@ti.data_oriented
class Model(AutoInit):
    def __init__(self, obj=None, tex=None):
        self.geo_list = []
        self.vertices = []
        self.L2W = Affine.var(())
        self.texture = None
        self.todo_obj = obj
        self.todo_tex = tex
        if obj is not None:
            vertex = Vertex.var(obj['v'].shape[0], has_tex=tex is not None)
            face = Face.var(obj['f'].shape[0])
            self.set_vertices(vertex)
            self.add_geometry(face)
        if tex is not None:
            assert tex.shape[2] == 3, "texture must be RGB"
            texture = ti.Vector.var(3, ti.f32, tex.shape[:2])
            self.set_texture(texture)

    def _init(self):
        self.L2W.init()
        if self.todo_obj is not None:
            self.vertices.pos.from_numpy(self.todo_obj['v'])
            self.geo_list[0].idx.from_numpy(self.todo_obj['f'])
        if self.todo_tex is not None:
            self.vertices.tex.from_numpy(self.todo_obj['vt'])
            self.texture.from_numpy(self.todo_tex.astype(np.float32) / 255)

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

    @ti.func
    def texSample(self, coor):
        return ts.bilerp(self.texture, coor * ts.vec(*self.texture.shape))

    def set_texture(self, text):
        self.texture = text
