from ..common import *


@ti.data_oriented
class MeshGrid:
    def __init__(self, res, as_quad=False):
        if isinstance(res, int):
            res = res, res
        self.res = V(*res)
        self.pos = ti.Vector.field(3, float, self.res)
        self.nrm = ti.Vector.field(3, float, self.res)
        self.tex = ti.Vector.field(2, float, self.res)
        self.as_quad = as_quad

        @ti.materialize_callback
        @ti.kernel
        def init_pos():
            for I in ti.grouped(self.pos):
                u, v = I / (self.res - 1)
                self.tex[I] = V(u, v)
                self.pos[I] = V(u * 2 - 1, v * 2 - 1, 0)

    def get_npolygon(self):
        return 4 if self.as_quad else 3

    @ti.func
    def pre_compute(self):
        for i, j in self.pos:
            i2 = max(i - 1, 0)
            j2 = max(j - 1, 0)
            i1 = min(i + 1, self.res.x - 1)
            j1 = min(j + 1, self.res.y - 1)
            dy = self.pos[i, j1] - self.pos[i, j2]
            dx = self.pos[i1, j] - self.pos[i2, j]
            self.nrm[i, j] = dx.cross(dy).normalized()

    @ti.func
    def get_nfaces(self):
        n = (self.res.x - 1) * (self.res.y - 1)
        if ti.static(self.as_quad):
            return n
        else:
            return n * 2

    @ti.func
    def _get_face_props(self, prop, n):
        stride = self.res.x - 1
        m = n
        if ti.static(not self.as_quad):
            m = n // 2
        i, j = V(m // stride, m % stride)
        a, b = prop[i, j], prop[i + 1, j]
        c, d = prop[i + 1, j + 1], prop[i, j + 1]
        if ti.static(self.as_quad):
            return a, b, c, d
        if n % 2 != 0:
            a, b, c = a, c, d
        return a, b, c

    @ti.func
    def get_face_verts(self, n):
        return self._get_face_props(self.pos, n)

    @ti.func
    def get_face_norms(self, n):
        return self._get_face_props(self.nrm, n)

    @ti.func
    def get_face_coors(self, n):
        return self._get_face_props(self.tex, n)
