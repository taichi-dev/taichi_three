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
class IndicedFace:
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


@ti.data_oriented
class MakeNormalFace:
    def __init__(self, face):
        self.face = face

    @property
    @ti.func
    def pos(self):
        return self.face.pos

    @property
    @ti.func
    def tex(self):
        return self.face.tex

    @property
    @ti.func
    def nrm(self):
        posa, posb, posc = self.pos
        normal = ts.cross(posa - posc, posa - posb)
        return normal, normal, normal


class Mesh:
    def __init__(self, faces, pos, tex, nrm):
        self.faces = faces
        self.pos = pos
        self.tex = pos
        self.nrm = nrm

    def loop_range(self):
        return self.faces.loop_range()

    @ti.func
    def get_face(self, i):
        return IndicedFace(self.faces[i], self.pos, self.tex, self.nrm)

    @classmethod
    def from_obj(cls, obj):
        faces = create_field((3, 3), int, len(obj['f']))
        pos = create_field(3, float, len(obj['vp']))
        tex = create_field(2, float, len(obj['vt']))
        nrm = create_field(3, float, len(obj['vn']))

        @ti.materialize_callback
        def init_mesh_data():
            faces.from_numpy(obj['f'])
            pos.from_numpy(obj['vp'])
            tex.from_numpy(obj['vt'])
            nrm.from_numpy(obj['vn'])

        mesh = cls(faces=faces, pos=pos, tex=tex, nrm=nrm)
        return mesh


class MeshMakeNormal:
    def __init__(self, mesh):
        self.mesh = mesh

    def loop_range(self):
        return self.mesh.loop_range()

    @ti.func
    def get_face(self, i):
        face = self.mesh.get_face(i)
        return MakeNormalFace(face)


class Model(ModelBase):
    def __init__(self, mesh):
        super().__init__()

        self.mesh = mesh

    @ti.func
    def render(self, camera):
        for i in ti.grouped(self.mesh):
            face = self.mesh.get_face(i)
            render_triangle(self, camera, face)

    @ti.func
    def intersect(self, orig, dir):
        hit = 1e6
        sorig, sdir = orig, dir
        clr = ts.vec3(0.0)
        for i in range(self.faces.shape[0]):
            face = IndicedFace(self.faces[i], self.pos, self.tex, self.nrm)
            ihit, iorig, idir, iclr = intersect_triangle(self, sorig, sdir, face)
            if ihit < hit:
                hit, orig, dir, clr = ihit, iorig, idir, iclr
        return hit, orig, dir, clr

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