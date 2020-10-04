import numpy as np
import taichi as ti
import taichi_glsl as ts
from .geometry import *
from .shading import *
from .transform import *
from .common import *
import math


@ti.data_oriented
class ModelBase(AutoInit):
    pass


class Model(ModelBase):
    def __init__(self, pos_n, tex_n, nrm_n, faces_n):
        self.L2W = Affine.field(())

        self.faces = ti.Matrix.field(3, 3, int, faces_n)
        self.pos = ti.Vector.field(3, float, pos_n)
        self.tex = ti.Vector.field(2, float, tex_n)
        self.nrm = ti.Vector.field(3, float, nrm_n)

        self.textures = {}
        self.init_cbs = []

    @classmethod
    def from_obj(cls, obj, texture=None, normtex=None):
        model = cls(len(obj['vp']), len(obj['vt']), len(obj['vn']), len(obj['f']))

        def obj_init_cb():
            model.faces.from_numpy(obj['f'])
            model.pos.from_numpy(obj['vp'])
            model.tex.from_numpy(obj['vt'])
            model.nrm.from_numpy(obj['vn'])

        model.init_cbs.append(obj_init_cb)

        if texture is not None:
            model.add_texture('color', texture)
        if normtex is not None:
            model.add_texture('normal', normtex)
        return model

    def add_texture(self, name, texture):
        assert name not in self.textures, name

        # convert UInt8 into Float32 for storage:
        if texture.dtype == np.uint8:
            texture = texture.astype(np.float32) / 255
        elif texture.dtype == np.float64:
            texture = texture.astype(np.float32)

        # normal maps are stored as [-1, 1] for maximizing FP precision:
        if name == 'normal':
            texture = texture * 2 - 1

        if len(texture.shape) == 3 and texture.shape[2] == 1:
            texture = texture.reshape(texture.shape[:2])

        # either RGB or greyscale
        if len(texture.shape) == 2:
            self.textures[name] = ti.field(float, texture.shape)

        else:
            assert len(texture.shape) == 3, texture.shape
            texture = texture[:, :, :3]
            assert texture.shape[2] == 3, texture.shape

            self.textures[name] = ti.Vector.field(3, float, texture.shape[:2])

        def other_init_cb():
            self.textures[name].from_numpy(texture)

        self.init_cbs.append(other_init_cb)

    def _init(self):
        self.L2W.init()
        for cb in self.init_cbs:
            cb()

    @ti.func
    def render(self, camera):
        for i in ti.grouped(self.faces):
            # assume all elements to be triangle
            render_triangle(self, camera, self.faces[i])

    @ti.func
    def sample(self, name: ti.template(), texcoor, default):
        if ti.static(name in self.textures.keys()):
            tex = ti.static(self.textures[name])
            return ts.bilerp(tex, texcoor * ts.vec(*tex.shape))
        else:
            return default

    def colorize(self, pos, texcoor, normal, color):
        roughness = self.sample('roughness', texcoor, CookTorrance.roughness)
        metallic = self.sample('metallic', texcoor, CookTorrance.metallic)
        opt = CookTorrance(roughness=roughness, metallic=metallic)
        opt.model = ti.static(self)
        return opt.colorize(pos, normal, color)

    @ti.func
    def pixel_shader(self, color, texcoor, normal):
        return color * self.sample('color', texcoor, ts.vec3(1.0))

    @ti.func
    def vertex_shader(self, pos, texcoor, normal, tangent, bitangent):
        color = ts.vec3(1.0)
        color = self.colorize(pos, texcoor, normal, color)
        return color, texcoor, normal


class ModelPP(Model):
    @ti.func
    def pixel_shader(self, pos, texcoor, normal, tangent, bitangent):
        ndir = self.sample('normal', texcoor, ts.vec3(0.0, 0.0, 1.0))
        normal = ti.Matrix.cols([tangent, bitangent, normal]) @ ndir
        normal = normal.normalized()

        color = self.sample('color', texcoor, ts.vec3(1.0))
        color = self.colorize(pos, texcoor, normal, color)
        return color

    @ti.func
    def vertex_shader(self, pos, texcoor, normal, tangent, bitangent):
        return pos, texcoor, normal, tangent, bitangent
