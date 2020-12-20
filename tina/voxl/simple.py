from ..common import *


@ti.data_oriented
class SimpleVolume:
    def __init__(self, N, coloring=False):
        self.coloring = coloring
        self.N = N

        if self.coloring:
            self.dens = ti.Vector.field(3, float, (N, N, N))
        else:
            self.dens = ti.field(float, (N, N, N))

        @ti.materialize_callback
        def init_pars():
            self.dens.fill(1)

    @ti.func
    def pre_compute(self):
        pass

    @ti.func
    def sample_volume(self, pos):
        return trilerp(self.dens, pos * self.N)
