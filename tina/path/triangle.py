from ..advans import *
from .geometry import *


@ti.data_oriented
class TriangleTracer:

    def __init__(self, maxfaces=65536*16, smoothing=False, texturing=False,
                 multimtl=True, **extra_options):
        self.smoothing = smoothing
        self.texturing = texturing
        self.multimtl = multimtl
        self.maxfaces = maxfaces

        self.verts = ti.Vector.field(3, float, (maxfaces, 3))
        if self.smoothing:
            self.norms = ti.Vector.field(3, float, (maxfaces, 3))
        if self.texturing:
            self.coors = ti.Vector.field(2, float, (maxfaces, 3))
        if self.multimtl:
            self.mtlids = ti.field(int, maxfaces)
        self.nfaces = ti.field(int, ())

    def clear_objects(self):
        self.nfaces[None] = 0

    @ti.kernel
    def add_object(self, mesh: ti.template(), mtlid: ti.template()):
        mesh.pre_compute()
        nfaces = mesh.get_nfaces()
        base = self.nfaces[None]
        self.nfaces[None] += nfaces
        for i in range(nfaces):
            j = base + i
            verts = mesh.get_face_verts(i)
            if ti.static(self.multimtl):
                self.mtlids[j] = mtlid
            for k in ti.static(range(3)):
                self.verts[j, k] = verts[k]
            if ti.static(self.smoothing):
                norms = mesh.get_face_norms(i)
                for k in ti.static(range(3)):
                    self.norms[j, k] = norms[k]
            if ti.static(self.texturing):
                coors = mesh.get_face_coors(i)
                for k in ti.static(range(3)):
                    self.coors[j, k] = coors[k]

    @ti.kernel
    def set_face_verts(self, verts: ti.ext_arr()):
        self.nfaces[None] = min(verts.shape[0], self.verts.shape[0])
        for i in range(self.nfaces[None]):
            for k in ti.static(range(3)):
                for l in ti.static(range(3)):
                    self.verts[i, k][l] = verts[i, k, l]

    @ti.kernel
    def set_face_norms(self, norms: ti.ext_arr()):
        ti.static_assert(self.smoothing)
        for i in range(self.nfaces[None]):
            for k in ti.static(range(3)):
                for l in ti.static(range(3)):
                    self.norms[i, k][l] = norms[i, k, l]

    @ti.kernel
    def set_face_coors(self, coors: ti.ext_arr()):
        ti.static_assert(self.texturing)
        for i in range(self.nfaces[None]):
            for k in ti.static(range(3)):
                for l in ti.static(range(2)):
                    self.coors[i, k][l] = coors[i, k, l]

    @ti.kernel
    def _export_vertices(self, verts: ti.ext_arr()):
        for i in range(self.nfaces[None]):
            for k in ti.static(range(3)):
                for l in ti.static(range(3)):
                    verts[i, k, l] = self.verts[i, k][l]

    def build(self, tree):
        verts = np.empty((self.nfaces[None], 3, 3))
        self._export_vertices(verts)
        bmax = np.max(verts, axis=1)
        bmin = np.min(verts, axis=1)
        tree.build(bmin, bmax)

    @ti.func
    def get_material_id(self, ind):
        if ti.static(not self.multimtl):
            return 0
        return self.mtlids[ind]

    @ti.func
    def calc_geometry(self, near, ind, uv, ro, rd):
        nrm = V(0., 0., 0.)
        tex = V(0., 0.)
        wei = V(1 - uv.x - uv.y, uv.x, uv.y)

        if ti.static(self.texturing):
            c0 = self.coors[ind, 0]
            c1 = self.coors[ind, 1]
            c2 = self.coors[ind, 2]
            tex = c0 * wei.x + c1 * wei.y + c2 * wei.z

        if ti.static(self.smoothing):
            n0 = self.norms[ind, 0]
            n1 = self.norms[ind, 1]
            n2 = self.norms[ind, 2]
            nrm = (n0 * wei.x + n1 * wei.y + n2 * wei.z).normalized()
        else:
            v0 = self.verts[ind, 0]
            v1 = self.verts[ind, 1]
            v2 = self.verts[ind, 2]
            nrm = (v1 - v0).cross(v2 - v0).normalized()

        return nrm, tex

    @ti.func
    def hit(self, ind, ro, rd):
        v0 = self.verts[ind, 0]
        v1 = self.verts[ind, 1]
        v2 = self.verts[ind, 2]
        hit, depth, uv = ray_triangle_hit(v0, v1, v2, ro, rd)
        return hit, depth, uv
