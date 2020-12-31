from ..advans import *
from .geom import *


@ti.data_oriented
class Particles:
    @ti.func
    def transmit(self, near, ind, uv, ro, rd, rc):
        ro += near * rd
        nrm = (ro - self.pos[ind]).normalized()

        if ind == 0:
            rc *= 2
            rd *= 0
        else:
            rd, wei = self.matr.sample(rd, nrm)
            rc *= wei

            '''
            if ti.random() < 0.5:
                lirad = 0.2
                lipos = spherical(ti.random() * 2 - 1, ti.random()) * lirad
                liarea = ti.pi * lirad**2 / 2
                toli = lipos - ro
                dis2 = toli.norm_sqr()
                toli = toli.normalized()
                if toli.dot(nrm) >= 0:
                    rc *= liarea / dis2
                    rd = toli
            '''

        ro += nrm * eps * 8
        return ro, rd, rc

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
