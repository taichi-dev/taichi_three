from ..advans import *
from .geometry import *


@ti.data_oriented
class ParticleTracer:  # TODO: realize me
    @ti.func
    def calc_geometry(self, near, ind, uv, ro, rd):
        nrm = (ro - self.pos[ind]).normalized()
        return nrm, V(0., 0.)

    def __init__(self, matr, pos, rad=0.05, dim=3):
        self.pos = ti.Vector.field(dim, float, len(pos))
        self.rad = rad

        @ti.materialize_callback
        def init_pos():
            self.pos.from_numpy(pos)

        self.matr = matr

    def build(self, tree):
        pos = self.pos.to_numpy()
        tree.build(pos - self.rad, pos + self.rad)

    @ti.func
    def hit(self, ind, ro, rd):
        pos = self.pos[ind]
        hit, depth = ray_sphere_hit(pos, self.rad, ro, rd)
        return hit, depth, V(0., 0.)