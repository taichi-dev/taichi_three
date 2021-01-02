from ..common import *
from .base import MeshEditBase


class MeshFlatNormal(MeshEditBase):
    @ti.func
    def _calc_norm(self, a, b, c):
        return (b - a).cross(c - a).normalized()

    @ti.func
    def get_face_norms(self, n):
        verts = self.mesh.get_face_verts(n)
        norm = self._calc_norm(verts[0], verts[1], verts[2])
        return [norm for vert in verts]


class MeshSmoothNormal(MeshEditBase):
    def __init__(self, mesh, cached=True):
        super().__init__(mesh)

        N = self.mesh.get_max_vert_nindex()
        self.norm = ti.Vector.field(3, float, N)

        self.cached = cached
        if self.cached:
            ti.materialize_callback(self.update_normal)

    @ti.func
    def pre_compute(self):
        self.mesh.pre_compute()
        if ti.static(not self.cached):
            self._update_normal()

    @ti.func
    def _update_normal(self):
        for i in self.norm:
            self.norm[i] = 0
        for n in range(self.mesh.get_nfaces()):
            i, j, k = self.mesh.get_face_vert_indices(n)
            a, b, c = self.mesh.get_face_verts(n)
            nrm = (b - a).cross(c - a).normalized()
            self.norm[i] += nrm
            self.norm[j] += nrm
            self.norm[k] += nrm
        for i in self.norm:
            self.norm[i] = self.norm[i].normalized()

    @ti.kernel
    def update_normal(self):
        self._update_normal()

    @ti.func
    def get_face_norms(self, n):
        i, j, k = self.mesh.get_face_vert_indices(n)
        return self.norm[i], self.norm[j], self.norm[k]
