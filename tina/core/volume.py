from ..common import *
from ..advans import *


@ti.data_oriented
class VolumeRaster:
    def __init__(self, engine, N=128, radius=None, gaussian=None, taa=False, **extra_options):
        self.engine = engine
        self.res = self.engine.res
        if radius is None:
            radius = 1 if taa else 12
        if gaussian is None:
            gaussian = not taa
        self.gaussian = gaussian
        self.radius = radius
        self.taa = taa
        self.N = N

        self.dens = ti.field(float, (N, N, N))
        self.occup = ti.field(float, self.res)
        self.tmcup = ti.field(float, self.res)

        @ti.materialize_callback
        def init_dens():
            self.dens.fill(1)

        if self.gaussian:
            self.wei = ti.field(float, self.radius + 1)

            @ti.materialize_callback
            @ti.kernel
            def init_wei():
                total = -1.0
                for i in self.wei:
                    x = 1.6 * i / (self.radius + 1)
                    r = ti.exp(-x**2)
                    self.wei[i] = r
                    total += r * 2
                for i in self.wei:
                    self.wei[i] /= total

    @ti.kernel
    def set_object(self, voxl: ti.template()):
        for I in ti.grouped(self.dens):
            self.dens[I] = voxl.sample_volume(I / self.N)

    @ti.kernel
    def render_occup(self):
        uniq = V(0, 0, 0).cast(ti.u32)
        if ti.static(self.taa):
            uniq = V(ti.random(ti.u32), ti.random(ti.u32), ti.random(ti.u32))
        for P in ti.grouped(self.occup):
            self.occup[P] = 0
        for I in ti.grouped(self.dens):
            bias = V(0., 0., 0.)
            if ti.static(self.taa):
                bias = ti.Vector([noise(V34(I, u)) for u in uniq])
            Pl = (I + bias) / self.N * 2 - 1
            Pv = self.engine.to_viewspace(Pl)
            if not all(-1 < Pv < 1):
                continue
            P = int(self.engine.to_viewport(Pv))
            if not all(0 <= P <= self.res - 1):
                continue

            DXl = mapply_dir(self.engine.V2W[None], V(1., 0., 0.)).normalized()
            DYl = mapply_dir(self.engine.V2W[None], V(0., 1., 0.)).normalized()
            Rv = V(0., 0.)
            Rl = 2 / self.N
            Rv.x = self.engine.to_viewspace(Pl + DXl * Rl).x - Pv.x
            Rv.y = self.engine.to_viewspace(Pl + DYl * Rl).y - Pv.y
            r = Rv * self.res

            rho = self.dens[I] / self.N
            self.occup[P] += rho * r.x * r.y / (2 * self.radius + 1)

    @ti.kernel
    def blur(self, src: ti.template(), dst: ti.template(), dir: ti.template()):
        for P in ti.grouped(src):
            res = 0.0
            bot = min(self.radius, P[dir])
            top = min(self.radius, self.res[dir] - 1 - P[dir])
            for i in range(-bot, top + 1):
                Q = P + U2(dir) * i
                fac = 1 / (top + 1 + bot)
                if ti.static(self.gaussian):
                    fac = self.wei[abs(i)]
                res += src[Q] * fac
            dst[P] = res

    @ti.kernel
    def render_color(self, shader: ti.template()):
        for P in ti.grouped(self.occup):
            rho = max(0, self.occup[P])
            shader.img[P] = 1 - ti.exp(-rho)

    def render(self, shader):
        self.render_occup()
        if self.radius:
            self.blur(self.occup, self.tmcup, 0)
            self.blur(self.tmcup, self.occup, 1)
        self.render_color(shader)
