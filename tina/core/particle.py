from ..common import *


@ti.data_oriented
class ParticleRaster:
    def __init__(self, engine, maxpars=MAX, coloring=True,
            clipping=True, **extra_options):
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
            self.sizes.fill(0.1)
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
    def set_particles(self, verts: ti.ext_arr()):
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
            vert = pars.get_particle_position(i)
            self.verts[i] = vert
            size = pars.get_particle_radius(i)
            self.sizes[i] = size
            if ti.static(self.coloring):
                color = pars.get_particle_color(i)
                self.colors[i] = color

    @ti.kernel
    def render_occup(self):
        for P in ti.grouped(self.occup):
            self.occup[P] = -1
        for f in ti.smart(self.get_particles_range()):
            Al = self.get_particle_position(f)
            Rl = self.get_particle_radius(f)
            Av = self.engine.to_viewspace(Al)
            if ti.static(self.clipping):
                if not -1 <= Av.z <= 1:
                    continue

            DXl = mapply_dir(self.engine.V2W[None], V(1., 0., 0.)).normalized()
            DYl = mapply_dir(self.engine.V2W[None], V(0., 1., 0.)).normalized()
            Rv = V(0., 0.)
            Rv.x = self.engine.to_viewspace(Al + DXl * Rl).x - Av.x
            Rv.y = self.engine.to_viewspace(Al + DYl * Rl).y - Av.y
            Bv = [
                    Av - V(Rv.x, 0., 0.),
                    Av + V(Rv.x, 0., 0.),
                    Av - V(0., Rv.y, 0.),
                    Av + V(0., Rv.y, 0.),
            ]
            a = self.engine.to_viewport(Av)
            b = [self.engine.to_viewport(Bv) for Bv in Bv]

            bot, top = ifloor(min(b[0], b[2])), iceil(max(b[1], b[3]))
            bot, top = max(bot, 0), min(top, self.res - 1)
            for P in ti.grouped(ti.ndrange((bot.x, top.x + 1), (bot.y, top.y + 1))):
                p = float(P) + self.engine.bias[None]
                Pv = V23(self.engine.from_viewport(p), Av.z)
                Pl = self.engine.from_viewspace(Pv)
                if (Pl - Al).norm_sqr() > Rl**2:
                    continue

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
            DXl = mapply_dir(self.engine.V2W[None], V(1., 0., 0.)).normalized()
            DYl = mapply_dir(self.engine.V2W[None], V(0., 1., 0.)).normalized()
            Rv = V(0., 0.)
            Rv.x = self.engine.to_viewspace(Al + DXl * Rl).x - Av.x
            Rv.y = self.engine.to_viewspace(Al + DYl * Rl).y - Av.y

            p = float(P) + self.engine.bias[None]
            Pv = V23(self.engine.from_viewport(p), Av.z)
            Pl = self.engine.from_viewspace(Pv)

            Dl = (Pl - Al) / Rl
            Zl = mapply_dir(self.engine.V2W[None], V(0., 0., 1.)).normalized()
            Dl -= Zl * ti.sqrt(1 - Dl.norm_sqr())
            Dl = Dl.normalized()

            normal = Dl
            pos = Al + Dl * Rl
            texcoord = V(0., 0.)
            color = self.get_particle_color(f)

            shader.shade_color(self.engine, P, p, f, pos, normal, texcoord, color)
