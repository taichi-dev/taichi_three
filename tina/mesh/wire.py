from ..common import *
from .base import MeshEditBase


class MeshToWire(MeshEditBase):
    def __init__(self, mesh):
        super().__init__(mesh)
        if hasattr(self.mesh, 'get_npolygon'):
            self.src_npolygon = self.mesh.get_npolygon()
        else:
            self.src_npolygon = 3

    def get_npolygon(self):
        return 2

    @ti.func
    def get_nfaces(self):
        return self.mesh.get_nfaces() * self.src_npolygon

    @ti.func
    def _get_face_props(self, src_getter: ti.template(), n):
        props = src_getter(n // self.src_npolygon)
        ep1 = n % self.src_npolygon
        ep2 = (n + 1) % self.src_npolygon
        p1 = list_subscript(props, ep1)
        p2 = list_subscript(props, ep2)
        return p1, p2

    @ti.func
    def get_face_verts(self, n):
        return self._get_face_props(self.mesh.get_face_verts, n)

    @ti.func
    def get_face_coors(self, n):
        return self._get_face_props(self.mesh.get_face_coors, n)

    @ti.func
    def get_face_norms(self, n):
        return self._get_face_props(self.mesh.get_face_norms, n)
