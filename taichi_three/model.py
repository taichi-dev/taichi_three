import numpy as np
import taichi as ti
import taichi_glsl as ts
from .geometry import *
from .transform import *
from .common import *
import math


@ti.data_oriented
class Model(AutoInit):
    def __init__(self, f_n=None, vi_n=None, vt_n=None, vn_n=None, obj=None):
        self.L2W = Affine.var(())

        self.faces = None
        self.vi = None
        self.vt = None
        self.vn = None

        if obj is not None:
            f_n = obj['f'].shape
            vi_n = obj['vi'].shape
            vt_n = obj['vt'].shape
            vn_n = obj['vn'].shape

        if f_n is not None:
            self.faces = ti.Vector.var(2, ti.i32, f_n)
        if vi_n is not None:
            self.vi = ti.Vector.var(2, ti.f32, vi_n)
        if vt_n is not None:
            self.vt = ti.Vector.var(1, ti.f32, vt_n)
        if vn_n is not None:
            self.vn = ti.Vector.var(2, ti.f32, vn_n)

        if obj is not None:
            self.init_obj = obj

    def from_obj(self, obj):
        self.faces.from_numpy(obj['f'])
        self.vi.from_numpy(obj['vi'])
        self.vt.from_numpy(obj['vt'])
        self.vn.from_numpy(obj['vn'])

    def _init(self):
        self.L2W.init()
        if hasattr(self, 'init_obj'):
            self.from_object(self.init_obj)

    @ti.func
    def render(self):
        self.geo.render()

    def set_geometry(geo):
        geo.model = self
        self.geo = geo

    @ti.func
    def texSample(self, coor):
        return ts.bilerp(self.texture, coor * ts.vec(*self.texture.shape))
