from ..common import *


@ti.data_oriented
class SimpleVolume:
    def __init__(self, N):
        self.N = N
        self.dens = ti.field(float, (N, N, N))

        @ti.materialize_callback
        def init_pars():
            self.dens.fill(1)

    def set_volume_density(self, dens):
        self.dens.from_numpy(dens)

    @ti.func
    def pre_compute(self):
        pass

    @ti.func
    def sample_volume(self, pos):
        ret = 0.0
        if all(0 <= pos <= 1):
            ret = trilerp(self.dens, pos * self.N)
        return ret

    @ti.func
    def sample_gradient(self, pos):
        ret = ti.Vector.zero(float, 3)
        for i in ti.static(range(3)):
            hi = self.sample_volume(pos + U3(i) / self.N)
            lo = self.sample_volume(pos - U3(i) / self.N)
            ret[i] = (hi - lo) / 2
        return ret

    @ti.func
    def get_transform(self):
        return ti.Matrix.identity(float, 4)
