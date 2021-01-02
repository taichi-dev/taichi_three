from ..advans import *
from .geometry import *


@ti.data_oriented
class PointLighting:
    def __init__(self, maxlights=16):
        self.maxlights = maxlights
        self.pos = ti.Vector.field(3, float, maxlights)
        self.color = ti.Vector.field(3, float, maxlights)
        self.rad = ti.field(float, maxlights)
        self.nlights = ti.field(int, ())

        @ti.materialize_callback
        def init_lights():
            self.nlights[None] = 1
            self.color.fill(1)
            self.rad.fill(0.1)

    @ti.kernel
    def set_lights(self, pos: ti.ext_arr()):
        self.nlights[None] = pos.shape[0]
        for i in range(self.nlights[None]):
            for k in ti.static(range(3)):
                self.pos[i][k] = pos[i, k]

    @ti.kernel
    def set_light_radii(self, rad: ti.ext_arr()):
        for i in range(self.nlights[None]):
            self.rad[i] = rad[i]

    @ti.kernel
    def set_light_colors(self, color: ti.ext_arr()):
        for i in range(self.nlights[None]):
            for k in ti.static(range(3)):
                self.color[i][k] = color[i, k]

    @ti.func
    def hit(self, ro, rd):
        ind = -1
        near = inf
        for i in range(self.nlights[None]):
            hit, depth = ray_sphere_hit(self.pos[i], self.rad[i], ro, rd)
            if hit != 0 and depth < near:
                ind = i
                near = depth
                break
        return near, ind, V(0., 0.)

    @ti.func
    def get_nlights(self):
        return self.nlights[None]

    @ti.func
    def redirect(self, ro, ind):
        lirad = self.rad[ind]
        lipos = self.pos[ind]
        lirip = spherical(ti.random() * 2 - 1, ti.random()) * lirad + lipos
        toli = lirip - ro
        dis2 = toli.norm_sqr()
        toli = toli.normalized()
        wei = self.color[ind] / dis2
        return toli, wei, ti.sqrt(dis2)