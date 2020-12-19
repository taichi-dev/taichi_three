from ..common import *


@ti.data_oriented
class MeshEditBase:
    def __init__(self, mesh):
        self.mesh = mesh

    @ti.func
    def pre_compute(self):
        self.mesh.pre_compute()

    @ti.func
    def get_nfaces(self):
        return self.mesh.get_nfaces()

    @ti.func
    def get_face_verts(self, n):
        verts = self.mesh.get_face_verts(n)
        return verts

    @ti.func
    def get_face_norms(self, n):
        norms = self.mesh.get_face_norms(n)
        return norms

    @ti.func
    def get_face_coors(self, n):
        coors = self.mesh.get_face_coors(n)
        return coors

    def __getattr__(self, attr):
        return getattr(self.mesh, attr)
