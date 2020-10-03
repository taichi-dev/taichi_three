import numpy as np
import taichi as ti
import taichi_glsl as ts
from .geometry import *
from .shading import *
from .transform import *
from .common import *
import math


@ti.data_oriented
class Model(AutoInit):
    def __init__(self, pos_n, tex_n, nrm_n, faces_n):
        self.L2W = Affine.field(())

        self.faces = ti.Matrix.field(3, 3, int, faces_n)
        self.pos = ti.Vector.field(3, float, pos_n)
        self.tex = ti.Vector.field(2, float, tex_n)
        self.nrm = ti.Vector.field(3, float, nrm_n)

        self.opt = CookTorrance()
        self.texture = None
        self.normtex = None

    @classmethod
    def from_obj(cls, obj, texture=None, normtex=None):
        model = cls(len(obj['vp']), len(obj['vt']), len(obj['vn']), len(obj['f']))

        def obj_init_cb():
            model.faces.from_numpy(obj['f'])
            model.pos.from_numpy(obj['vp'])
            model.tex.from_numpy(obj['vt'])
            model.nrm.from_numpy(obj['vn'])

        model.obj_init_cb = obj_init_cb

        model.load_texture(texture, normtex)
        return model

    def load_texture(self, texture=None, normtex=None):
        if texture is not None:
            if texture.dtype == np.uint8:
                texture = texture.astype(np.float32) / 255
            assert len(texture.shape) == 3
            texture = texture[:, :, :3]
            assert texture.shape[2] == 3

        if normtex is not None:
            if normtex.dtype == np.uint8:
                normtex = normtex.astype(np.float32) / 255
                normtex = normtex * 2 - 1
            assert len(normtex.shape) == 3
            normtex = normtex[:, :, :3]
            assert normtex.shape[2] == 3

        if texture is not None:
            self.texture = ti.Vector(3, float, texture.shape[:2])
        if normtex is not None:
            self.normtex = ti.Vector(3, float, normtex.shape[:2])

        def other_init_cb():
            if texture is not None:
                self.texture.from_numpy(texture)
            if normtex is not None:
                self.normtex.from_numpy(normtex)

        self.other_init_cb = other_init_cb

    def _init(self):
        self.L2W.init()
        self.obj_init_cb()
        self.other_init_cb()

    def obj_init_cb(self):
        pass

    def other_init_cb(self):
        pass

    @ti.func
    def render(self, camera):
        for i in ti.grouped(self.faces):
            # assume all elements to be triangle
            render_triangle(self, camera, self.faces[i])

    @ti.func
    def texSample(self, texcoor):
        if ti.static(self.texture is not None):
            return ts.bilerp(self.texture, texcoor * ts.vec(*self.texture.shape))
        else:
            return ts.vec3(1.0)

    @ti.func
    def nrmtexSample(self, texcoor):
        if ti.static(self.normtex is not None):
            return ts.bilerp(self.normtex, texcoor * ts.vec(*self.normtex.shape))
        else:
            return ts.vec3(0.0, 0.0, 1.0)

    @ti.func
    def pixel_shader(self, color, texcoor, normal):
        return color * self.texSample(texcoor)

    @ti.func
    def vertex_shader(self, pos, texcoor, normal, tangent, bitangent):
        color = ts.vec3(1.0)
        color = self.opt.colorize(self.scene, pos, normal, color)
        return color, texcoor, normal


class ModelPP(Model):
    @ti.func
    def pixel_shader(self, pos, texcoor, normal, tangent, bitangent):
        ndir = self.nrmtexSample(texcoor)
        normal = ti.Matrix.cols([tangent, bitangent, normal]) @ ndir
        normal = normal.normalized()
        color = self.texSample(texcoor)
        color = self.opt.colorize(self.scene, pos, normal, color)
        return color

    @ti.func
    def vertex_shader(self, pos, texcoor, normal, tangent, bitangent):
        return pos, texcoor, normal, tangent, bitangent
