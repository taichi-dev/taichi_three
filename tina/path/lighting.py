from ..advans import *
from .geometry import *


@ti.data_oriented
class RTXLighting:
    def __init__(self, maxlights=16):
        self.maxlights = maxlights
        self.pos = ti.Vector.field(3, float, maxlights)
        self.color = ti.Vector.field(3, float, maxlights)
        self.rad = ti.field(float, maxlights)
        self.nlights = ti.field(int, ())
        self.skybox = tina.Skybox('assets/grass.jpg')

        @ti.materialize_callback
        def init_lights():
            #self.nlights[None] = 1
            self.color.fill(2.0)
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

    @ti.func
    def background(self, rd):
        if ti.static(hasattr(self, 'skybox')):
            return self.skybox.sample(rd)
        else:
            return 0.0

    @ti.func
    def shade_color(self, material, tree, pos, normal, viewdir):
        N_li, N_sky = 8, 8

        res = V(0.0, 0.0, 0.0)
        ro = pos + normal * eps * 8
        for lind in range(self.get_nlights()):
            for s in range(N_li):
                # cast shadow ray to lights
                ldir, lwei, ldis = self.redirect(pos, lind)
                lwei *= max(0, ldir.dot(normal))
                if Vall(lwei <= 0):
                    continue
                occdis, occind, occuv = tree.hit(ro, ldir)
                if occdis < ldis:  # shadow occlusion
                    continue
                lwei *= material.brdf(normal, ldir, viewdir)
                res += lwei / N_li

        if ti.static(hasattr(self, 'skybox')):
            for s in range(N_sky):
                ldir, lwei = material.sample(viewdir, normal, 1)
                occdis, occind, occuv = tree.hit(ro, ldir)
                if occdis >= inf:
                    lclr = lwei * self.background(ldir)
                    res += lclr / N_sky

        return res
