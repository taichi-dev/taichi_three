from ..common import *


@ti.data_oriented
class DynamicMesh:
    def __init__(self, maxfaces=65536, npolygon=3):
        self.verts = ti.Vector.field(3, float, (maxfaces, npolyon))
        self.coors = ti.Vector.field(2, float, (maxfaces, npolyon))
        self.norms = ti.Vector.field(3, float, (maxfaces, npolyon))
        self.nfaces = ti.field(float, ())

        self.maxfaces = maxfaces
        self.npolygon = npolygon

    @ti.func
    def pre_compute(self):
        pass

    @ti.func
    def get_nfaces(self):
        return min(self.nfaces[None], self.maxfaces)

    @ti.func
    def _get_face_props(self, prop, n):
        return [prop[n, i] for i in range(self.npolygon)]

    @ti.func
    def get_face_verts(self, n):
        return self._get_face_props(self.verts, n)

    @ti.func
    def get_face_coors(self, n):
        return self._get_face_props(self.coors, n)

    @ti.func
    def get_face_norms(self, n):
        return self._get_face_props(self.norms, n)

    @ti.kernel
    def set_face_verts(self, verts: ti.ext_arr()):
        self.nfaces[None] = min(verts.shape[0], self.verts.shape[0])
        for i in range(self.nfaces[None]):
            for k in ti.static(range(3)):
                for l in ti.static(range(3)):
                    self.verts[i, k][l] = verts[i, k, l]

    @ti.kernel
    def set_face_norms(self, norms: ti.ext_arr()):
        for i in range(self.nfaces[None]):
            for k in ti.static(range(3)):
                for l in ti.static(range(3)):
                    self.norms[i, k][l] = norms[i, k, l]

    @ti.kernel
    def set_face_coors(self, coors: ti.ext_arr()):
        for i in range(self.nfaces[None]):
            for k in ti.static(range(3)):
                for l in ti.static(range(2)):
                    self.coors[i, k][l] = coors[i, k, l]
