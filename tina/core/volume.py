from ..common import *


@ti.data_oriented
class VolumeRaster:
    def __init__(self, engine, N=32, coloring=True, **extra_options):
        self.engine = engine
        self.res = self.engine.res
        self.N = N

        self.dens = ti.Vector.field(3, float, (N, N, N))
        self.occup = ti.field(float, self.res)

    @ti.kernel
    def render_occup(self):
        for P in ti.grouped(self.occup):
            self.occup[P] = 0
        for I in ti.grouped(self.dens):
            Pl = (I / self.res) * 2 - 1
            Pv = self.engine.to_viewspace(Pl)
            P = self.engine.to_viewport(Pv)

            self.occup[P] += self.dens[I]

    @ti.kernel
    def render_color(self, shader: ti.template()):
        for P in ti.grouped(self.occup):
            rho = self.occup[P]
            if rho == 0:
                continue

            shader.img[P] = 1

    def render(self, shader):
        self.render_occup()
        self.render_color(shader)
