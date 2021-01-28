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
        if all(-1 <= pos <= 1):
            ret = trilerp(self.dens, (pos * 0.5 + 0.5) * self.N)
        return ret

    @ti.func
    def sample_gradient(self, pos):
        ret = ti.Vector.zero(float, 3)
        for i in ti.static(range(3)):
            dir = U3(i) * 0.5 / self.N
            hi = self.sample_volume(pos + dir)
            lo = self.sample_volume(pos - dir)
            ret[i] = (hi - lo) / 2
        return ret

    @ti.func
    def get_transform(self):
        return ti.Matrix.identity(float, 4)

    def get_bounding_box(self):
        return V3(-1.), V3(1.)
