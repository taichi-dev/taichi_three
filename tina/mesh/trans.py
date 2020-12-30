from ..common import *
from .base import MeshEditBase


class MeshTransform(MeshEditBase):
    def __init__(self, mesh, trans=None):
        super().__init__(mesh)

        self.trans = ti.Matrix.field(4, 4, float, ())
        self.trans_normal = ti.Matrix.field(3, 3, float, ())

        @ti.materialize_callback
        @ti.kernel
        def init_trans():
            self.trans[None] = ti.Matrix.identity(float, 4)
            self.trans_normal[None] = ti.Matrix.identity(float, 3)

        if trans is not None:
            @ti.materialize_callback
            def init_trans_arg():
                self.set_transform(trans)

    def set_transform(self, trans):
        trans_normal = np.transpose(np.linalg.inv(trans))
        self.trans[None] = np.array(trans).tolist()
        self.trans_normal[None] = np.array(trans_normal).tolist()

    @ti.func
    def get_face_verts(self, n):
        verts = self.mesh.get_face_verts(n)
        for i, vert in ti.static(enumerate(verts)):
            verts[i] = mapply_pos(self.trans[None], vert)
        return verts

    @ti.func
    def get_face_norms(self, n):
        norms = self.mesh.get_face_norms(n)
        for i, norm in ti.static(enumerate(norms)):
            norms[i] = self.trans_normal[None] @ norm
        return norms
