from ..common import *


@ti.data_oriented
class TriangleRaster:
    def __init__(self, engine, maxfaces=MAX, smoothing=False, texturing=False,
            culling=True, clipping=True, **extra_options):
        self.engine = engine
        self.res = self.engine.res
        self.maxfaces = maxfaces
        self.smoothing = smoothing
        self.texturing = texturing
        self.culling = culling
        self.clipping = clipping

        self.occup = ti.field(int, self.res)

        self.nfaces = ti.field(int, ())
        self.verts = ti.Vector.field(3, float, (maxfaces, 3))
        if self.smoothing:
            self.norms = ti.Vector.field(3, float, (maxfaces, 3))
        if self.texturing:
            self.coors = ti.Vector.field(2, float, (maxfaces, 3))

        self.bcn = ti.Vector.field(2, float, maxfaces)
        self.can = ti.Vector.field(2, float, maxfaces)
        self.boo = ti.Vector.field(2, float, maxfaces)
        self.coo = ti.Vector.field(2, float, maxfaces)
        self.wsc = ti.Vector.field(3, float, maxfaces)

    @ti.func
    def interpolate(self, shader: ti.template(), P, p, f, wei, A, B, C):
        pos = wei.x * A + wei.y * B + wei.z * C

        normal = V(0., 0., 0.)
        if ti.static(self.smoothing):
            An, Bn, Cn = self.get_face_normals(f)
            normal = wei.x * An + wei.y * Bn + wei.z * Cn
        else:
            normal = (B - A).cross(C - A)  # let the shader normalize it
        normal = normal.normalized()

        texcoord = V(0., 0.)
        if ti.static(self.texturing):
            At, Bt, Ct = self.get_face_texcoords(f)
            texcoord = wei.x * At + wei.y * Bt + wei.z * Ct

        color = V(1., 1., 1.)
        shader.shade_color(self.engine, P, p, f, pos, normal, texcoord, color)

    @ti.func
    def get_faces_range(self):
        for i in range(self.nfaces[None]):
            yield i

    @ti.func
    def get_face_vertices(self, f):
        A, B, C = self.verts[f, 0], self.verts[f, 1], self.verts[f, 2]
        return A, B, C

    @ti.func
    def get_face_normals(self, f):
        A, B, C = self.norms[f, 0], self.norms[f, 1], self.norms[f, 2]
        return A, B, C

    @ti.func
    def get_face_texcoords(self, f):
        A, B, C = self.coors[f, 0], self.coors[f, 1], self.coors[f, 2]
        return A, B, C

    @ti.kernel
    def set_object(self, mesh: ti.template()):
        mesh.pre_compute()
        self.nfaces[None] = mesh.get_nfaces()
        for i in range(self.nfaces[None]):
            verts = mesh.get_face_verts(i)
            for k in ti.static(range(3)):
                self.verts[i, k] = verts[k]
            if ti.static(self.smoothing):
                norms = mesh.get_face_norms(i)
                for k in ti.static(range(3)):
                    self.norms[i, k] = norms[k]
            if ti.static(self.texturing):
                coors = mesh.get_face_coors(i)
                for k in ti.static(range(3)):
                    self.coors[i, k] = coors[k]

    @ti.kernel
    def render_occup(self):
        for P in ti.grouped(self.occup):
            self.occup[P] = -1
        for f in ti.smart(self.get_faces_range()):
            Al, Bl, Cl = self.get_face_vertices(f)
            Av, Bv, Cv = [self.engine.to_viewspace(p) for p in [Al, Bl, Cl]]
            facing = (Bv.xy - Av.xy).cross(Cv.xy - Av.xy)
            if facing <= 0:
                if ti.static(self.culling):
                    continue

            if ti.static(self.clipping):
                if not all(-1 <= Av <= 1):
                    if not all(-1 <= Bv <= 1):
                        if not all(-1 <= Cv <= 1):
                            continue

            a, b, c = [self.engine.to_viewport(p) for p in [Av, Bv, Cv]]

            bot, top = ifloor(min(a, b, c)), iceil(max(a, b, c))
            bot, top = max(bot, 0), min(top, self.res - 1)
            n = (b - a).cross(c - a)
            bcn = (b - c) / n
            can = (c - a) / n
            wscale = 1 / ti.Vector([mapply(self.engine.W2V[None], p, 1)[1] for p in [Al, Bl, Cl]])
            for P in ti.grouped(ti.ndrange((bot.x, top.x + 1), (bot.y, top.y + 1))):
                pos = float(P) + self.engine.bias[None]
                w_bc = (pos - b).cross(bcn)
                w_ca = (pos - c).cross(can)
                wei = V(w_bc, w_ca, 1 - w_bc - w_ca) * wscale
                wei /= wei.x + wei.y + wei.z
                if all(wei >= 0):
                    depth_f = wei.x * Av.z + wei.y * Bv.z + wei.z * Cv.z
                    depth = int(depth_f * self.engine.maxdepth)
                    if ti.atomic_min(self.engine.depth[P], depth) > depth:
                        if self.engine.depth[P] >= depth:
                            self.occup[P] = f

            self.bcn[f] = bcn
            self.can[f] = can
            self.boo[f] = b
            self.coo[f] = c
            self.wsc[f] = wscale

    @ti.kernel
    def render_color(self, shader: ti.template()):
        for P in ti.grouped(self.occup):
            f = self.occup[P]
            if f == -1:
                continue

            Al, Bl, Cl = self.get_face_vertices(f)

            bcn = self.bcn[f]
            can = self.can[f]
            b = self.boo[f]
            c = self.coo[f]
            wscale = self.wsc[f]
            p = float(P) + self.engine.bias[None]
            w_bc = (p - b).cross(bcn)
            w_ca = (p - c).cross(can)
            wei = V(w_bc, w_ca, 1 - w_bc - w_ca) * wscale
            wei /= wei.x + wei.y + wei.z

            self.interpolate(shader, P, p, f, wei, Al, Bl, Cl)
