from ..common import *
from .base import MeshEditBase


class FlatNormal(MeshEditBase):
    @ti.func
    def _calc_norm(self, a, b, c):
        return (b - a).cross(c - a).normalized()

    @ti.func
    def get_face_norms(self, n):
        verts = self.mesh.get_face_verts(n)
        norm = self._calc_norm(verts[0], verts[1], verts[2])
        return [norm for vert in verts]


# TODO: SmoothNormal requires getting f2v connections from mesh prototype
