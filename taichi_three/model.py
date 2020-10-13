import numpy as np
import taichi as ti
import taichi_glsl as ts
from .geometry import *
from .shading import *
from .transform import *
from .common import *
import math


@ti.data_oriented
class ModelBase:
    def __init__(self):
        self.L2W = ti.Matrix.field(4, 4, float, ())
        self.L2C = ti.Matrix.field(4, 4, float, ())

        @ti.materialize_callback
        @ti.kernel
        def init_L2W():
            self.L2W[None] = ti.Matrix.identity(float, 4)
            self.L2C[None] = ti.Matrix.identity(float, 4)

        self.init_cbs = []

    @ti.func
    def set_view(self, camera):
        self.L2C[None] = camera.L2W[None].inverse() @ self.L2W[None]


class ModelLow(ModelBase):
    def __init__(self, faces_n, pos_n, tex_n, nrm_n):
        super().__init__()

        if not hasattr(self, 'faces'):
            self.faces = ti.Matrix.field(3, 3, int, faces_n)
        if not hasattr(self, 'pos'):
            self.pos = ti.Vector.field(3, float, pos_n)
        if not hasattr(self, 'tex'):
            self.tex = ti.Vector.field(2, float, tex_n)
        if not hasattr(self, 'nrm'):
            self.nrm = ti.Vector.field(3, float, nrm_n)

        self.textures = {}
        self.material = Material(CookTorrance())

    @ti.func
    def render(self, camera):
        for i in ti.grouped(self.faces):
            # assume all elements to be triangle
            render_triangle(self, camera, self.faces[i])

    @ti.func
    def intersect(self, orig, dir):
        hit = 1e6
        sorig, sdir = orig, dir
        clr = ts.vec3(0.0)
        for i in range(self.faces.shape[0]):
            ihit, iorig, idir, iclr = intersect_triangle(self, sorig, sdir, self.faces[i])
            if ihit < hit:
                hit, orig, dir, clr = ihit, iorig, idir, iclr
        return hit, orig, dir, clr

    @classmethod
    def from_obj(cls, obj, texture=None, normtex=None):
        model = cls(len(obj['f']), len(obj['vp']), len(obj['vt']), len(obj['vn']))

        @ti.materialize_callback
        def init_mesh_data():
            model.faces.from_numpy(obj['f'])
            model.pos.from_numpy(obj['vp'])
            model.tex.from_numpy(obj['vt'])
            model.nrm.from_numpy(obj['vn'])

        if texture is not None:
            model.add_texture('color', texture)
        if normtex is not None:
            model.add_texture('normal', normtex)
        return model

    def add_uniform(self, name, value):
        self.add_texture(name, np.array([[value]]))

    @ti.func
    def sample(self, name: ti.template(), texcoor, default):
        if ti.static(name in self.textures.keys()):
            tex = ti.static(self.textures[name])
            return ts.bilerp(tex, texcoor * ts.vec(*tex.shape))
        else:
            return default

    def radiance(self, pos, indir, texcoor, normal):
        # TODO: we don't support normal maps in path tracing mode for now
        with self.material.specify_inputs(model=self, pos=pos, texcoor=texcoor, normal=normal, tangent=normal, bitangent=normal, indir=indir) as shader:
            return shader.radiance()

    def colorize(self, pos, texcoor, normal, tangent, bitangent):
        with self.material.specify_inputs(model=self, pos=pos, texcoor=texcoor, normal=normal, tangent=tangent, bitangent=bitangent) as shader:
            return shader.colorize()

    @ti.func
    def pixel_shader(self, pos, color, texcoor, normal):
        color = color * self.sample('color', texcoor, ts.vec3(1.0))
        return dict(img=color, pos=pos, normal=normal)

    @ti.func
    def vertex_shader(self, pos, texcoor, normal, tangent, bitangent):
        color = self.colorize(pos, texcoor, normal, tangent, bitangent)
        return pos, color, texcoor, normal


class Model(ModelLow):
    @ti.func
    def pixel_shader(self, pos, texcoor, normal, tangent, bitangent):
        # normal has been no longer normalized due to lerp and ndir errors.
        # so here we re-enforce normalization to get slerp.
        normal = normal.normalized()
        color = self.colorize(pos, texcoor, normal, tangent, bitangent)
        return dict(img=color, pos=pos, texcoor=texcoor, normal=normal,
                    tangent=tangent, bitangent=bitangent)

    @ti.func
    def vertex_shader(self, pos, texcoor, normal, tangent, bitangent):
        return pos, texcoor, normal, tangent, bitangent