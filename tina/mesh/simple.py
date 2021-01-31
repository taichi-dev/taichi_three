from ..common import *


@ti.data_oriented
class SimpleMesh:
    def __init__(self, maxfaces=MAX, npolygon=3):
        '''
        :param maxfaces: (int) the maximum amount of faces to be supported
        :param npolygon: (int) number of polygon edges, 3 for triangles
        :return: (Mesh) the mesh object to add into scene
        '''

        self.verts = ti.Vector.field(3, float, (maxfaces, npolygon))
        self.coors = ti.Vector.field(2, float, (maxfaces, npolygon))
        self.norms = ti.Vector.field(3, float, (maxfaces, npolygon))
        self.mtlids = ti.field(int, maxfaces)
        self.nfaces = ti.field(int, ())

        self.maxfaces = maxfaces
        self.npolygon = npolygon

    def get_npolygon(self):
        return self.npolygon

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

    @ti.func
    def get_face_mtlid(self, n):
        return self.mtlids[n]

    @ti.kernel
    def set_face_verts(self, verts: ti.ext_arr()):
        '''
        :param verts: (np.array[nfaces, npolygon, 3]) the vertex positions of faces

        Specify the face vertex positions to be rendered

        :note: the number of faces is determined by the array's shape[0]
        '''

        self.nfaces[None] = min(verts.shape[0], self.verts.shape[0])
        for i in range(self.nfaces[None]):
            for k in ti.static(range(self.npolygon)):
                for l in ti.static(range(3)):
                    self.verts[i, k][l] = verts[i, k, l]

    @ti.kernel
    def set_face_norms(self, norms: ti.ext_arr()):
        '''
        :param norms: (np.array[nfaces, npolygon, 3]) the vertex normal vectors of faces

        Specify the face vertex normals to be rendered

        :note: the normals should be normalized to get desired result
        :note: this should be invoked only *after* set_face_verts for nfaces
        '''

        for i in range(self.nfaces[None]):
            for k in ti.static(range(self.npolygon)):
                for l in ti.static(range(3)):
                    self.norms[i, k][l] = norms[i, k, l]

    @ti.kernel
    def set_face_coors(self, coors: ti.ext_arr()):
        '''
        :param norms: (np.array[nfaces, npolygon, 2]) the vertex texture coordinates of faces

        Specify the face vertex texcoords to be rendered

        :note: this should be invoked only *after* set_face_verts for nfaces
        '''

        for i in range(self.nfaces[None]):
            for k in ti.static(range(self.npolygon)):
                for l in ti.static(range(2)):
                    self.coors[i, k][l] = coors[i, k, l]

    @ti.kernel
    def set_face_mtlids(self, mtlids: ti.ext_arr()):
        '''
        :param mtlids: (np.array[nfaces]) the material ids of faces

        Specify the face material ids to be rendered

        :note: this should be invoked only *after* set_face_verts for nfaces
        '''
        for i in range(self.nfaces[None]):
            self.mtlids[i] = mtlids[i]

    @ti.kernel
    def set_material_id(self, mtlid: int):
        for i in range(self.nfaces[None]):
            self.mtlids[i] = mtlid
