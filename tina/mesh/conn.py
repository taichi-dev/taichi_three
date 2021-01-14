from ..common import *


@ti.data_oriented
class ConnectiveMesh:
    def __init__(self, maxfaces=MAX, maxverts=MAX, npolygon=3):
        self.faces = ti.Vector.field(npolygon, int, maxfaces)
        self.verts = ti.Vector.field(3, float, maxverts)
        self.coors = ti.Vector.field(2, float, maxverts)
        self.norms = ti.Vector.field(3, float, maxverts)
        self.nfaces = ti.field(int, ())

        self.maxfaces = maxfaces
        self.maxverts = maxverts
        self.npolygon = npolygon

    def get_npolygon(self):
        return self.npolygon

    @ti.func
    def pre_compute(self):
        pass

    def get_max_vert_nindex(self):
        return self.maxverts

    @ti.func
    def get_indiced_vert(self, i):
        return self.verts[i]

    @ti.func
    def get_indiced_norm(self, i):
        return self.norms[i]

    @ti.func
    def get_indiced_coor(self, i):
        return self.coors[i]

    @ti.func
    def get_nfaces(self):
        return self.nfaces[None]

    @ti.func
    def get_face_vert_indices(self, n):
        i = self.faces[n][0]
        j = self.faces[n][1]
        k = self.faces[n][2]
        return i, j, k

    @ti.func
    def _get_face_props(self, prop, n):
        a = prop[self.faces[n][0]]
        b = prop[self.faces[n][1]]
        c = prop[self.faces[n][2]]
        return a, b, c

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
    def set_vertices(self, verts: ti.ext_arr()):
        nverts = min(verts.shape[0], self.verts.shape[0])
        for i in range(nverts):
            for k in ti.static(range(3)):
                self.verts[i][k] = verts[i, k]

    @ti.kernel
    def set_vert_norms(self, norms: ti.ext_arr()):
        nverts = min(norms.shape[0], self.norms.shape[0])
        for i in range(nverts):
            for k in ti.static(range(3)):
                self.norms[i][k] = norms[i, k]

    @ti.kernel
    def set_vert_coors(self, coors: ti.ext_arr()):
        nverts = min(coors.shape[0], self.coors.shape[0])
        for i in range(nverts):
            for k in ti.static(range(3)):
                self.coors[i][k] = coors[i, k]

    @ti.kernel
    def set_faces(self, faces: ti.ext_arr()):
        self.nfaces[None] = min(faces.shape[0], self.faces.shape[0])
        for i in range(self.nfaces[None]):
            for k in ti.static(range(self.npolygon)):
                self.faces[i][k] = faces[i, k]

