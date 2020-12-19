from ..common import *


@ti.data_oriented
class TriangleRaster:
    def __init__(self, engine, maxfaces=65536, smoothing=False, texturing=False,
            culling=True, clipping=False):
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

    @ti.func
    def interpolate(self, shader: ti.template(), P, f, facing, wei, A, B, C):
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

        if ti.static(not self.culling):
            if facing < 0:
                normal = -normal
        shader.shade_color(self.engine, P, f, pos, normal, texcoord)

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
            for P in ti.grouped(ti.ndrange((bot.x, top.x + 1), (bot.y, top.y + 1))):
                pos = float(P) + self.engine.bias[None]
                w_bc = (pos - b).cross(bcn)
                w_ca = (pos - c).cross(can)
                wei = V(w_bc, w_ca, 1 - w_bc - w_ca)
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
            pos = float(P) + self.engine.bias[None]
            w_bc = (pos - b).cross(bcn)
            w_ca = (pos - c).cross(can)
            wei = V(w_bc, w_ca, 1 - w_bc - w_ca)
            wei /= V(*[mapply(self.engine.W2V[None], p, 1)[1] for p in [Al, Bl, Cl]])
            wei /= wei.x + wei.y + wei.z

            self.interpolate(shader, P, f, 1, wei, Al, Bl, Cl)

    def render(self, shader):
        self.render_occup()
        self.render_color(shader)


@ti.data_oriented
class ParticleRaster:
    def __init__(self, engine, maxpars=65536, coloring=False, clipping=True):
        self.engine = engine
        self.res = self.engine.res
        self.maxpars = maxpars
        self.coloring = coloring
        self.clipping = clipping

        self.occup = ti.field(int, self.res)

        self.npars = ti.field(int, ())
        self.verts = ti.Vector.field(3, float, maxpars)
        self.sizes = ti.field(float, maxpars)
        if self.coloring:
            self.colors = ti.Vector.field(3, float, maxpars)

        @ti.materialize_callback
        def init_pars():
            self.sizes.fill(0.05)
            if self.coloring:
                self.colors.fill(1)

    @ti.func
    def get_particles_range(self):
        for i in range(self.npars[None]):
            yield i

    @ti.func
    def get_particle_position(self, f):
        return self.verts[f]

    @ti.func
    def get_particle_radius(self, f):
        return self.sizes[f]

    @ti.func
    def get_particle_color(self, f):
        return self.colors[f]

    @ti.kernel
    def set_particle_positions(self, verts: ti.ext_arr()):
        self.npars[None] = min(verts.shape[0], self.verts.shape[0])
        for i in range(self.npars[None]):
            for k in ti.static(range(3)):
                self.verts[i][k] = verts[i, k]

    @ti.kernel
    def set_particle_radii(self, sizes: ti.ext_arr()):
        for i in range(self.npars[None]):
            self.sizes[i] = sizes[i]

    @ti.kernel
    def set_particle_colors(self, colors: ti.ext_arr()):
        ti.static_assert(self.coloring)
        for i in range(self.npars[None]):
            for k in ti.static(range(3)):
                self.colors[i][k] = colors[i, k]

    @ti.kernel
    def set_object(self, pars: ti.template()):
        pars.pre_compute()
        self.npars[None] = pars.get_npars()
        for i in range(self.npars[None]):
            vert = mesh.get_particle_position(i)
            self.verts[i] = vert
            size = mesh.get_particle_radius(i)
            self.sizes[i] = size
            if ti.static(self.coloring):
                color = mesh.get_particle_color(i)
                self.colors[i] = color

    @ti.kernel
    def render_occup(self):
        for P in ti.grouped(self.occup):
            self.occup[P] = -1
        for f in ti.smart(self.get_particles_range()):
            Al = self.get_particle_position(f)
            Rl = self.get_particle_radius(f)
            Av = self.engine.to_viewspace(Al)
            Rv = self.engine.to_viewspace_scalar(Al, Rl)
            if ti.static(self.clipping):
                if not all(-1 - Rv <= Av <= 1 + Rv):
                    continue

            a = self.engine.to_viewport(Av)
            r = self.engine.to_viewport_scalar(Rv)

            bot, top = ifloor(a - r), iceil(a + r)
            bot, top = max(bot, 0), min(top, self.res - 1)
            for P in ti.grouped(ti.ndrange((bot.x, top.x + 1), (bot.y, top.y + 1))):
                pos = float(P) + self.engine.bias[None]

                dpos = pos - a.xy
                dp2 = dpos.norm_sqr()
                if dp2 > r**2:
                    continue

                #dz = ti.sqrt(r**2 - dp2)
                #nrm = V23(dpos, dz).normalized()
                #depth_f = Av.z - max(nrm.z, 0) * Rv  ##
                depth_f = Av.z
                depth = int(depth_f * self.engine.maxdepth)
                if ti.atomic_min(self.engine.depth[P], depth) > depth:
                    if self.engine.depth[P] >= depth:
                        self.occup[P] = f

    @ti.kernel
    def render_color(self, shader: ti.template()):
        for P in ti.grouped(self.occup):
            f = self.occup[P]
            if f == -1:
                continue

            Al = self.get_particle_position(f)
            Rl = self.get_particle_radius(f)
            Av = self.engine.to_viewspace(Al)
            Rv = self.engine.to_viewspace_scalar(Al, Rl)
            a = self.engine.to_viewport(Av)
            r = self.engine.to_viewport_scalar(Rv)

            dpos = float(P) + self.engine.bias[None] - a.xy
            dz = ti.sqrt(r**2 - dpos.norm_sqr())

            NV2W = linear_part(self.engine.W2V[None]).transpose()
            nrm = (NV2W @ V23(dpos, -dz)).normalized()

            pos = Al
            normal = nrm
            texcoord = V(0., 0.)
            shader.shade_color(self.engine, P, f, pos, normal, texcoord)

    def render(self, shader):
        self.render_occup()
        self.render_color(shader)


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

    @ti.func
    def draw_ball(self, a, r):
        bot, top = ifloor(a - r), iceil(a + r)
        bot, top = max(bot, 0), min(top, self.res - 1)
        for i, j in ti.ndrange((bot.x, top.x + 1), (bot.y, top.y + 1)):
            dpos = float(V(i, j, a.z)) - a
            dp2 = dpos.norm_sqr()
            if dp2 > r**2:
                continue

            dz = ti.sqrt(r**2 - dp2)
            n = V23(dpos.xy, -dz)
            nrm = ts.normalize(n)
            yield a + nrm * r, nrm

    def __init__(self, res=512):

        self.res = tovector((res, res) if isinstance(res, int) else res)

        self.depth = ti.field(int, self.res)
        self.maxdepth = 2**30

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
    def to_viewspace_scalar(self, p, r):
        w = mapply(self.W2V[None], p, 1)[1]
        return r / w

    @ti.func
    def to_viewport(self, p):
        return (p.xy * 0.5 + 0.5) * self.res

    @ti.func
    def to_viewport_scalar(self, r):
        avgsize = (self.res.x + self.res.y) / 2
        return r * avgsize  ##

    @ti.kernel
    def clear_depth(self):
        for P in ti.grouped(self.depth):
            self.depth[P] = self.maxdepth

    def set_camera(self, view, proj):
        W2V = proj @ view
        V2W = np.linalg.inv(W2V)
        self.W2V.from_numpy(np.array(W2V, dtype=np.float32))
        self.V2W.from_numpy(np.array(V2W, dtype=np.float32))
