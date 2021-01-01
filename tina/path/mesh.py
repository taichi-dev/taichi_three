from ..advans import *
from .geom import *


@ti.data_oriented
class Triangles:
    @ti.func
    def get_normal(self, near, ind, uv, ro, rd):
        v0 = self.verts[ind, 0]
        v1 = self.verts[ind, 1]
        v2 = self.verts[ind, 2]
        nrm = (v1 - v0).cross(v2 - v0).normalized()
        if nrm.dot(rd) > 0:
            nrm = -nrm
        return nrm

    def __init__(self, matr, verts):
        self.verts = ti.Vector.field(3, float, (len(verts), 3))

        @ti.materialize_callback
        def init_verts():
            self.verts.from_numpy(verts)

        self.matr = matr

    def build(self, tree):
        verts = self.verts.to_numpy()
        bmax = np.max(verts, axis=1)
        bmin = np.min(verts, axis=1)
        tree.build(bmin, bmax)

    @ti.func
    def hit(self, ind, ro, rd):
        v0 = self.verts[ind, 0]
        v1 = self.verts[ind, 1]
        v2 = self.verts[ind, 2]
        depth, uv = ray_triangle_hit(v0, v1, v2, ro, rd)
        hit = 1 if depth < inf else 0
        return hit, depth, uv
