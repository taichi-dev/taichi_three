from ..advans import *
from .geom import *


@ti.data_oriented
class Particles:
    @ti.func
    def get_normal(self, near, ind, uv, ro, rd):
        return (ro - self.pos[ind]).normalized()

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