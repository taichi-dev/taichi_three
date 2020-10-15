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

        self.material = Material(CookTorrance())

    @ti.func
    def set_view(self, camera):
        self.L2C[None] = camera.L2W[None].inverse() @ self.L2W[None]


@ti.data_oriented
class IndicedTriangle:
    def __init__(self, face, postab, textab, nrmtab):
        self.face = face
        self.postab = postab
        self.textab = textab
        self.nrmtab = nrmtab

    @property
    @ti.func
    def pos(self):
        return self.postab[self.face[0, 0]], self.postab[self.face[1, 0]], self.postab[self.face[2, 0]]

    @property
    @ti.func
    def tex(self):
        return self.textab[self.face[0, 1]], self.textab[self.face[1, 1]], self.textab[self.face[2, 1]]

    @property
    @ti.func
    def nrm(self):
        return self.nrmtab[self.face[0, 2]], self.nrmtab[self.face[1, 2]], self.nrmtab[self.face[2, 2]]


class Model(ModelBase):
    def __init__(self, faces, pos, tex, nrm):
        super().__init__()

        self.faces = faces
        self.pos = pos
        self.tex = pos
        self.nrm = nrm

    @ti.func
    def render(self, camera):
        for i in ti.grouped(self.faces):
            # assume all elements to be triangle
            face = IndicedTriangle(self.faces[i], self.pos, self.tex, self.nrm)
            render_triangle(self, camera, face)

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
    def from_obj(cls, obj):
        faces = create_field((3, 3), int, len(obj['f']))
        pos = create_field(3, float, len(obj['vp']))
        tex = create_field(2, float, len(obj['vt']))
        nrm = create_field(3, float, len(obj['vn']))

        model = cls(faces=faces, pos=pos, tex=tex, nrm=nrm)

        @ti.materialize_callback
        def init_mesh_data():
            faces.from_numpy(obj['f'])
            pos.from_numpy(obj['vp'])
            tex.from_numpy(obj['vt'])
            nrm.from_numpy(obj['vn'])

        return model

    def radiance(self, pos, indir, texcoor, normal, tangent, bitangent):
        # TODO: we don't support normal maps in path tracing mode for now
        with self.material.specify_inputs(model=self, pos=pos, texcoor=texcoor, normal=normal, tangent=tangent, bitangent=bitangent, indir=indir) as shader:
            return shader.radiance()

    def colorize(self, pos, texcoor, normal, tangent, bitangent):
        with self.material.specify_inputs(model=self, pos=pos, texcoor=texcoor, normal=normal, tangent=tangent, bitangent=bitangent) as shader:
            return shader.colorize()

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