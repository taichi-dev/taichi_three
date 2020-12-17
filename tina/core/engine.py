from ..common import *


@ti.data_oriented
class Engine:
    @ti.func
    def draw_line(self, src, dst):
        dlt = dst - src
        adlt = abs(dlt)
        k, siz = V(1.0, 1.0), 0
        if adlt.x >= adlt.y:
            k.x = 1.0 if dlt.x >= 0 else -1.0
            k.y = k.x * dlt.y / dlt.x
            siz = int(adlt.x)
        else:
            k.y = 1.0 if dlt.y >= 0 else -1.0
            k.x = k.y * dlt.x / dlt.y
            siz = int(adlt.y)
        for i in range(siz + 1):
            pos = src + k * i
            yield pos, i / siz

    @ti.func
    def draw_trip(self, a, b, c):
        bot, top = ifloor(min(a, b, c)), iceil(max(a, b, c))
        bot, top = max(bot, 0), min(top, self.res - 1)
        n = (b - a).cross(c - a)
        bcn = (b - c) / n
        can = (c - a) / n
        for i, j in ti.ndrange((bot.x, top.x + 1), (bot.y, top.y + 1)):
            pos = float(V(i, j)) + 0.5
            w_bc = (pos - b).cross(bcn)
            w_ca = (pos - c).cross(can)
            wei = V(w_bc, w_ca, 1 - w_bc - w_ca)
            if all(wei >= 0):
                yield pos, wei

    def __init__(self, res=512, maxfaces=65536,
            smoothing=False, texturing=False, culling=True, clipping=False):

        self.res = tovector((res, res) if isinstance(res, int) else res)
        self.culling = culling
        self.clipping = clipping
        self.smoothing = smoothing
        self.texturing = texturing

        self.depth = ti.field(int, self.res)
        self.occup = ti.field(int, self.res)
        self.maxdepth = 2**30

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

        self.W2V = ti.Matrix.field(4, 4, float, ())
        self.V2W = ti.Matrix.field(4, 4, float, ())

        self.bias = ti.Vector.field(2, float, ())

        @ti.materialize_callback
        @ti.kernel
        def init_engine():
            self.W2V[None] = ti.Matrix.identity(float, 4)
            self.W2V[None][2, 2] = -1
            self.V2W[None] = ti.Matrix.identity(float, 4)
            self.V2W[None][2, 2] = -1
            self.bias[None] = [0.5, 0.5]

        ti.materialize_callback(self.clear_depth)

    @ti.kernel
    def randomize_bias(self, center: ti.template()):
        if ti.static(center):
            self.bias[None] = [0.5, 0.5]
        else:
            #r = ti.sqrt(ti.random())
            #a = ti.random() * ti.tau
            #x, y = r * ti.cos(a) * 0.5 + 0.5, r * ti.sin(a) * 0.5 + 0.5
            x, y = ti.random(), ti.random()
            self.bias[None] = [x, y]

    @ti.func
    def to_viewspace(self, p):
        return mapply_pos(self.W2V[None], p)

    @ti.func
    def to_viewport(self, p):
        return (p.xy * 0.5 + 0.5) * self.res

    @ti.kernel
    def render_occup(self):
        for P in ti.grouped(self.occup):
            self.occup[P] = -1
        for f in ti.smart(self.get_faces_range()):
            Al, Bl, Cl = self.get_face_vertices(f)
            Av, Bv, Cv = [self.to_viewspace(p) for p in [Al, Bl, Cl]]
            facing = (Bv.xy - Av.xy).cross(Cv.xy - Av.xy)
            if facing <= 0:
                if ti.static(self.culling):
                    continue

            if ti.static(self.clipping):
                if not all(-1 <= Av <= 1):
                    if not all(-1 <= Bv <= 1):
                        if not all(-1 <= Cv <= 1):
                            continue

            a, b, c = [self.to_viewport(p) for p in [Av, Bv, Cv]]

            bot, top = ifloor(min(a, b, c)), iceil(max(a, b, c))
            bot, top = max(bot, 0), min(top, self.res - 1)
            n = (b - a).cross(c - a)
            bcn = (b - c) / n
            can = (c - a) / n
            for i, j in ti.ndrange((bot.x, top.x + 1), (bot.y, top.y + 1)):
                pos = float(V(i, j)) + self.bias[None]
                w_bc = (pos - b).cross(bcn)
                w_ca = (pos - c).cross(can)
                wei = V(w_bc, w_ca, 1 - w_bc - w_ca)
                if all(wei >= 0):
                    P = int(pos)
                    depth_f = wei.x * Av.z + wei.y * Bv.z + wei.z * Cv.z
                    depth = int(depth_f * self.maxdepth)
                    if ti.atomic_min(self.depth[P], depth) > depth:
                        if self.depth[P] >= depth:
                            self.occup[P] = f

            self.bcn[f] = bcn
            self.can[f] = can
            self.boo[f] = b
            self.coo[f] = c

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
            pos = float(P) + self.bias[None]
            w_bc = (pos - b).cross(bcn)
            w_ca = (pos - c).cross(can)
            wei = V(w_bc, w_ca, 1 - w_bc - w_ca)
            wei /= V(*[mapply(self.W2V[None], p, 1)[1] for p in [Al, Bl, Cl]])
            wei /= wei.x + wei.y + wei.z

            self.interpolate(shader, P, f, 1, wei, Al, Bl, Cl)

    def render(self, shader):
        self.render_occup()
        self.render_color(shader)

    @ti.kernel
    def clear_depth(self):
        for P in ti.grouped(self.depth):
            self.depth[P] = self.maxdepth

    @ti.func
    def interpolate(self, shader: ti.template(), P, f, facing, wei, A, B, C):
        pos = wei.x * A + wei.y * B + wei.z * C

        normal = V(0., 0., 0.)
        if ti.static(self.smoothing):
            An, Bn, Cn = self.get_face_normals(f)
            normal = wei.x * An + wei.y * Bn + wei.z * Cn
        else:
            normal = (B - A).cross(C - A)  # let the shader normalize it

        texcoord = V(0., 0.)
        if ti.static(self.texturing):
            At, Bt, Ct = self.get_face_texcoords(f)
            texcoord = wei.x * At + wei.y * Bt + wei.z * Ct

        if ti.static(not self.culling):
            if facing < 0:
                normal = -normal
        shader.shade_color(self, P, f, pos, normal, texcoord)

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

    def set_camera(self, view, proj):
        W2V = proj @ view
        V2W = np.linalg.inv(W2V)
        self.W2V.from_numpy(np.array(W2V, dtype=np.float32))
        self.V2W.from_numpy(np.array(V2W, dtype=np.float32))
