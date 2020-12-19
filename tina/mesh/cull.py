from ..common import *
from .base import MeshEditBase


class MeshFlipCulling(MeshEditBase):
    @ti.func
    def _flip(self, props: ti.template()):
        props_rev = list(reversed(props))
        for i, prop in ti.static(enumerate(props_rev)):
            props[i] = prop

    @ti.func
    def get_face_verts(self, n):
        verts = self.mesh.get_face_verts(n)
        self._flip(verts)
        return verts

    @ti.func
    def get_face_norms(self, n):
        norms = self.mesh.get_face_norms(n)
        self._flip(norms)
        return norms

    @ti.func
    def get_face_coors(self, n):
        coors = self.mesh.get_face_coors(n)
        self._flip(coors)
        return coors


class MeshNoCulling(MeshFlipCulling):
    @ti.func
    def get_nfaces(self):
        return self.mesh.get_nfaces() * 2

    @ti.func
    def get_face_verts(self, n):
        verts = self.mesh.get_face_verts(n // 2)
        if n % 2 != 0:
            self._flip(verts)
        return verts

    @ti.func
    def get_face_norms(self, n):
        norms = self.mesh.get_face_norms(n // 2)
        if n % 2 != 0:
            self._flip(norms)
            for i, norm in ti.static(enumerate(norms)):
                norms[i] = -norm
        return norms

    @ti.func
    def get_face_coors(self, n):
        coors = self.mesh.get_face_coors(n // 2)
        if n % 2 != 0:
            self._flip(coors)
        return coors


class MeshFlipNormal(MeshEditBase):
    @ti.func
    def get_face_norms(self, n):
        norms = self.mesh.get_face_norms(n)
        for i, norm in ti.static(enumerate(norms)):
            norms[i] = -norm
        return norms
