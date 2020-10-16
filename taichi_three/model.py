import numpy as np
import taichi as ti
import taichi_glsl as ts
from .geometry import *
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

    @ti.func
    def set_view(self, camera):
        self.L2C[None] = camera.L2W[None].inverse() @ self.L2W[None]


@ti.data_oriented
class IndicedFace:
    def __init__(self, face, postab, textab, nrmtab, mid):
        self.face = face
        self.postab = postab
        self.textab = textab
        self.nrmtab = nrmtab
        self.mid = mid

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
class Mesh:
    def __init__(self, faces, pos, tex, nrm, mid=1):
        self.faces = faces
        self.pos = pos
        self.tex = tex
        self.nrm = nrm
        self.mid = mid

    @property
    def shape(self):
        return self.faces.shape

    @property
    def static_shape(self):
        return []

    def before_rendering(self):
        pass

    @ti.func
    def get_face(self, i, j: ti.template()):
        return IndicedFace(self.faces[i], self.pos, self.tex, self.nrm, self.mid[i])

    @classmethod
    def from_obj(cls, obj):
        if isinstance(obj, str):
            from .loader import readobj
            obj = readobj(obj)

        faces = create_field((3, 3), int, len(obj['f']))
        pos = create_field(3, float, len(obj['vp']))
        tex = create_field(2, float, len(obj['vt']))
        nrm = create_field(3, float, len(obj['vn']))
        mid = create_field((), int, len(obj['fm']))
        assert len(obj['f']) == len(obj['fm'])

        @ti.materialize_callback
        def init_mesh_data():
            faces.from_numpy(obj['f'])
            pos.from_numpy(obj['vp'])
            tex.from_numpy(obj['vt'])
            nrm.from_numpy(obj['vn'])
            mid.from_numpy(obj['fm'])

        mesh = cls(faces=faces, pos=pos, tex=tex, nrm=nrm, mid=mid)
        return mesh


@ti.data_oriented
class MeshMakeNormal:
    @ti.data_oriented
    class MakeNormalFace:
        def __init__(self, face):
            self.face = face

        @property
        @ti.func
        def mid(self):
            return self.face.mid

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

    def __init__(self, mesh):
        self.mesh = mesh

    @property
    def shape(self):
        return self.mesh.shape

    @property
    def static_shape(self):
        return self.mesh.static_shape

    def before_rendering(self):
        self.mesh.before_rendering()

    @ti.func
    def get_face(self, i, j: ti.template()):
        face = self.mesh.get_face(i, j)
        return self.MakeNormalFace(face)


@ti.data_oriented
class Model(ModelBase):
    def __init__(self, mesh, mid=1):
        super().__init__()
        self.mesh = mesh

    @ti.func
    def render(self, camera):
        self.mesh.before_rendering()
        for i in ti.grouped(ti.ndrange(*self.mesh.shape)):
            for j in ti.static(ti.grouped(ti.ndrange(*self.mesh.static_shape))):
                face = self.mesh.get_face(i, j)
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


@ti.data_oriented
class MeshGrid:
    @ti.data_oriented
    class MeshGridFace:
        def __init__(self, parent, i):
            self.parent = parent
            self.i = i

        @property
        @ti.func
        def mid(self):
            return self.parent.mid

        @property
        @ti.func
        def pos(self):
            return [self.parent.pos[i] for i in [self.i + ts.D.__, self.i + ts.D.x_, self.i + ts.D.xx, self.i + ts.D._x]]

        @property
        @ti.func
        def tex(self):
            return [i / ts.vec2(*self.parent.res) for i in [self.i + ts.D.__, self.i + ts.D.x_, self.i + ts.D.xx, self.i + ts.D._x]]

        @property
        @ti.func
        def nrm(self):
            return [self.parent.snrm[i] for i in [self.i + ts.D.__, self.i + ts.D.x_, self.i + ts.D.xx, self.i + ts.D._x]]

    def __init__(self, res, mid=1):
        super().__init__()
        self.res = res
        self.pos = ti.Vector.field(3, float, self.res)
        self.snrm = ti.Vector.field(3, float, self.res)
        self.mid = mid

        @ti.materialize_callback
        @ti.kernel
        def init_pos():
            for i in ti.grouped(self.pos):
                self.pos[i] = ts.vec(i / ts.vec(*self.pos.shape) * 2 - 1, 0.0).xzy

    @ti.func
    def before_rendering(self):
        for i in ti.grouped(self.snrm):
            self.snrm[i] = self.get_normal_at(i)

    @ti.func
    def get_normal_at(self, i):
        xa = self.pos[ts.clamp(i + ts.D.x_, 0, ts.vec2(*self.shape))]
        xb = self.pos[ts.clamp(i + ts.D.X_, 0, ts.vec2(*self.shape))]
        ya = self.pos[ts.clamp(i + ts.D._x, 0, ts.vec2(*self.shape))]
        yb = self.pos[ts.clamp(i + ts.D._X, 0, ts.vec2(*self.shape))]
        return (ya - yb).cross(xa - xb).normalized()

    @property
    def shape(self):
        return [self.res[0] - 1, self.res[1] - 1]

    @property
    def static_shape(self):
        return []

    @ti.func
    def get_face(self, i, j: ti.template()):
        return self.MeshGridFace(self, i)


@ti.data_oriented
class QuadToTri:
    def __init__(self, mesh):
        self.mesh = mesh

    @property
    def shape(self):
        return self.mesh.shape

    @property
    def static_shape(self):
        return [2, *self.mesh.static_shape]

    def before_rendering(self):
        self.mesh.before_rendering()

    @ti.func
    def get_face(self, i, j: ti.template()):
        face = self.mesh.get_face(i, ts.vec(*[j[_] for _ in range(1, j.n)]))
        ret = DataOriented()
        if ti.static(j.x == 0):
            ret.__dict__.update(
                mid = face.mid,
                pos = [face.pos[i] for i in [0, 1, 2]],
                tex = [face.tex[i] for i in [0, 1, 2]],
                nrm = [face.nrm[i] for i in [0, 1, 2]],
            )
        else:
            ret.__dict__.update(
                mid = face.mid,
                pos = [face.pos[i] for i in [0, 2, 3]],
                tex = [face.tex[i] for i in [0, 2, 3]],
                nrm = [face.nrm[i] for i in [0, 2, 3]],
            )
        return ret