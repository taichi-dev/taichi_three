from ..common import *


@ti.data_oriented
class SimpleVolume:
    def __init__(self, N):
        self.dens = ti.field(float, (N, N, N))

        @ti.materialize_callback
        def init_pars():
            self.dens.fill(1)

        self.N = N

    @ti.func
    def pre_compute(self):
        pass

    @ti.func
    def sample_volume(self, pos):
        return trilerp(self.dens, pos * self.N)
