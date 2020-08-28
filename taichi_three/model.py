import numpy as np
import taichi as ti
import taichi_glsl as ts
from .geometry import *
from .transform import *
from .common import *
import math


@ti.data_oriented
class Model(AutoInit):
    def __init__(self, f_n=None, f_m=None,
            vi_n=None, vt_n=None, vn_n=None, tex_n=None,
            obj=None, tex=None):
        self.L2W = Affine.field(())

        self.faces = None
        self.vi = None
        self.vt = None
        self.vn = None
        self.tex = None

        if obj is not None:
            f_n = None if obj['f'] is None else obj['f'].shape[0]
            vi_n = None if obj['vi'] is None else obj['vi'].shape[0]
            vt_n = None if obj['vt'] is None else obj['vt'].shape[0]
            vn_n = None if obj['vn'] is None else obj['vn'].shape[0]

        if tex is not None:
            tex_n = tex.shape[:2]

        if f_m is None:
            f_m = 1
            if vt_n is not None:
                f_m = 2
            if vn_n is not None:
                f_m = 3

        if vi_n is None:
            vi_n = 1
        if vt_n is None:
            vt_n = 1
        if vn_n is None:
            vn_n = 1

        if f_n is not None:
            self.faces = ti.Matrix.field(3, f_m, ti.i32, f_n)
        if vi_n is not None:
            self.vi = ti.Vector.field(3, ti.f32, vi_n)
        if vt_n is not None:
            self.vt = ti.Vector.field(2, ti.f32, vt_n)
        if vn_n is not None:
            self.vn = ti.Vector.field(3, ti.f32, vn_n)
        if tex_n is not None:
            self.tex = ti.Vector.field(3, ti.f32, tex_n)

        if obj is not None:
            self.init_obj = obj
        if tex is not None:
            self.init_tex = tex

    def from_obj(self, obj):
        if obj['f'] is not None:
            self.faces.from_numpy(obj['f'])
        if obj['vi'] is not None:
            self.vi.from_numpy(obj['vi'])
        if obj['vt'] is not None:
            self.vt.from_numpy(obj['vt'])
        if obj['vn'] is not None:
            self.vn.from_numpy(obj['vn'])

    def _init(self):
        self.L2W.init()
        if hasattr(self, 'init_obj'):
            self.from_obj(self.init_obj)
        if hasattr(self, 'init_tex'):
            self.tex.from_numpy(self.init_tex.astype(np.float32) / 255)

    @ti.func
    def render(self, camera):
        for i in ti.grouped(self.faces):
            render_triangle(self, camera, self.faces[i])

    @ti.func
    def texSample(self, coor):
        if ti.static(self.tex is not None):
            return ts.bilerp(self.tex, coor * ts.vec(*self.tex.shape))
        else:
            return 1
