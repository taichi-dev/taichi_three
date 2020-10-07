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
    def __init__(self):
        self.L2W = Affine.field(())
        self.init_cbs = []

    def _init(self):
        self.L2W.init()
        for cb in self.init_cbs:
            cb()


class ModelLow(ModelBase):
    def __init__(self, faces_n, pos_n, tex_n, nrm_n):
        super().__init__()

        self.faces = ti.Matrix.field(3, 3, int, faces_n)
        self.pos = ti.Vector.field(3, float, pos_n)
        self.tex = ti.Vector.field(2, float, tex_n)
        self.nrm = ti.Vector.field(3, float, nrm_n)

        self.textures = {}
        self.shading_type = CookTorrance

    @ti.func
    def render(self, camera):
        for i in ti.grouped(self.faces):
            # assume all elements to be triangle
            render_triangle(self, camera, self.faces[i])

    @ti.func
    def intersect(self, camera, I, orig, dir):
        for i in range(self.faces.shape[0]):
            intersect_triangle(self, camera, I, orig, dir, self.faces[i])

    @classmethod
    def from_obj(cls, obj, texture=None, normtex=None):
        model = cls(len(obj['f']), len(obj['vp']), len(obj['vt']), len(obj['vn']))

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

    def add_uniform(self, name, value):
        self.add_texture(name, np.array([[value]]))

    @ti.func
    def sample(self, name: ti.template(), texcoor, default):
        if ti.static(name in self.textures.keys()):
            tex = ti.static(self.textures[name])
            return ts.bilerp(tex, texcoor * ts.vec(*tex.shape))
        else:
            return default

    def colorize(self, pos, texcoor, normal):
        opt = self.shading_type()
        opt.model = self
        for key in opt.parameters:
            setattr(opt, key, self.sample(key, texcoor, getattr(opt, key)))
        return opt.colorize(pos, normal)

    @ti.func
    def pixel_shader(self, pos, color, texcoor, normal):
        color = color * self.sample('color', texcoor, ts.vec3(1.0))
        return dict(img=color, pos=pos, normal=normal)

    @ti.func
    def vertex_shader(self, pos, texcoor, normal, tangent, bitangent):
        color = self.colorize(pos, texcoor, normal)
        return pos, color, texcoor, normal


class SimpleModel(ModelBase):
    def __init__(self, faces_n, pos_n):
        super().__init__()

        self.pos = ti.Vector.field(3, float, pos_n)
        self.clr = ti.Vector.field(3, float, pos_n)
        self.faces = ti.Vector.field(3, int, faces_n)

        @ti.materialize_callback
        def initialize_clr():
            self.clr.fill(1.0)

    @ti.func
    def render(self, camera):
        for i in ti.grouped(self.faces):
            face = ti.Matrix.cols([self.faces[i], self.faces[i], ts.vec3(0)])
            render_triangle(self, camera, face)

    @subscriptable
    @ti.func
    def tex(self, I):
        return self.clr[I]

    @subscriptable
    def nrm(self, I):
        return ts.vec3(0.0, 0.0, -1.0)

    @ti.func
    def pixel_shader(self, pos, color):
        return dict(img=color, pos=pos)

    @ti.func
    def vertex_shader(self, pos, texcoor, normal, tangent, bitangent):
        color = texcoor
        return pos, color


class Model(ModelLow):
    @ti.func
    def pixel_shader(self, pos, texcoor, normal, tangent, bitangent):
        ndir = self.sample('normal', texcoor, ts.vec3(0.0, 0.0, 1.0))
        normal = ti.Matrix.cols([tangent, bitangent, normal]) @ ndir
        # normal has been no longer normalized due to lerp and ndir errors.
        # so here we re-enforce normalization to get slerp.
        normal = normal.normalized()

        color = self.colorize(pos, texcoor, normal)
        return dict(img=color, pos=pos, normal=normal,
                    tangent=tangent, bitangent=bitangent)

    @ti.func
    def vertex_shader(self, pos, texcoor, normal, tangent, bitangent):
        return pos, texcoor, normal, tangent, bitangent