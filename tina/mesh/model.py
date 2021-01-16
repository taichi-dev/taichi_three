from ..common import *


@ti.data_oriented
class MeshModel:
    def __init__(self, obj, *_, **__):
        '''
        :param obj: (OBJ | str) the OBJ model as to be displayed, automatically invoke tina.readobj if a string is specified
        :return: (Mesh) the mesh object to add into scene
        '''

        if isinstance(obj, str):
            obj = tina.readobj(obj, *_, **__)

        self.faces = ti.Matrix.field(3, 3, int, len(obj['f']))
        self.verts = ti.Vector.field(3, float, len(obj['v']))
        self.coors = ti.Vector.field(2, float, len(obj['vt']))
        self.norms = ti.Vector.field(3, float, len(obj['vn']))

        @ti.materialize_callback
        def init_mesh():
            faces = obj['f']
            if len(faces.shape) == 2:
                faces = np.stack([faces, faces, faces], axis=2)
            self.faces.from_numpy(faces.astype(np.uint32))
            self.verts.from_numpy(obj['v'])
            self.coors.from_numpy(obj['vt'])
            self.norms.from_numpy(obj['vn'])

        self.maxfaces = len(obj['f'])
        self.maxverts = len(obj['v'])
        self.maxcoors = len(obj['vt'])
        self.maxnorms = len(obj['vn'])

    def get_npolygon(self):
        return 3

    @ti.func
    def pre_compute(self):
        pass

    def get_max_vert_nindex(self):
        return self.maxverts

    @ti.func
    def get_nfaces(self):
        return self.faces.shape[0]

    @ti.func
    def get_face_vert_indices(self, n):
        i = self.faces[n][0, 0]
        j = self.faces[n][1, 0]
        k = self.faces[n][2, 0]
        return i, j, k

    @ti.func
    def _get_face_props(self, prop, index: ti.template(), n):
        a = prop[self.faces[n][0, index]]
        b = prop[self.faces[n][1, index]]
        c = prop[self.faces[n][2, index]]
        return a, b, c

    @ti.func
    def get_face_verts(self, n):
        return self._get_face_props(self.verts, 0, n)

    @ti.func
    def get_face_coors(self, n):
        return self._get_face_props(self.coors, 1, n)

    @ti.func
    def get_face_norms(self, n):
        return self._get_face_props(self.norms, 2, n)
