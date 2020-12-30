from ..common import *


@ti.data_oriented
class SimpleParticles:
    def __init__(self, maxpars=65536, radius=0.02):
        self.verts = ti.Vector.field(3, float, maxpars)
        self.sizes = ti.field(float, maxpars)
        self.colors = ti.Vector.field(3, float, maxpars)
        self.npars = ti.field(int, ())

        @ti.materialize_callback
        def init_pars():
            self.sizes.fill(radius)
            self.colors.fill(1)

        self.maxpars = maxpars

    @ti.func
    def pre_compute(self):
        pass

    @ti.func
    def get_npars(self):
        return min(self.npars[None], self.maxpars)

    @ti.func
    def get_particle_position(self, n):
        return self.verts[n]

    @ti.func
    def get_particle_radius(self, n):
        return self.sizes[n]

    @ti.func
    def get_particle_color(self, n):
        return self.colors[n]

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
        for i in range(self.npars[None]):
            for k in ti.static(range(3)):
                self.colors[i][k] = colors[i, k]
