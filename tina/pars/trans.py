from ..common import *
from .base import ParsEditBase


class ParsTransform(ParsEditBase):
    def __init__(self, pars):
        super().__init__(pars)

        self.trans = ti.Matrix.field(4, 4, float, ())
        self.scale = ti.field(float, ())

        @ti.materialize_callback
        @ti.kernel
        def init_trans():
            self.trans[None] = ti.Matrix.identity(float, 4)
            self.scale[None] = 1

    def set_transform(self, trans, scale):
        self.trans[None] = np.array(trans).tolist()
        self.scale[None] = scale

    @ti.func
    def get_particle_position(self, n):
        vert = self.pars.get_particle_position(n)
        return mapply_pos(self.trans[None], vert)

    @ti.func
    def get_particle_radius(self, n):
        size = self.pars.get_particle_radius(n)
        return self.scale[None] * size
