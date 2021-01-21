from ..advans import *
from .geometry import *


@ti.data_oriented
class PathLighting:
    def __init__(self, maxlights=16):
        self.maxlights = maxlights
        self.pos = ti.Vector.field(3, float, maxlights)
        self.color = ti.Vector.field(3, float, maxlights)
        self.rad = ti.field(float, maxlights)
        self.nlights = ti.field(int, ())

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
    def emit_light(self, ind):
        dir = spherical(ti.random() * 2 - 1, ti.random())
        ro = dir * self.rad[ind] + self.pos[ind]
        rd = tangentspace(dir) @ spherical(ti.random(), ti.random())
        return ro, rd

    @ti.func
    def redirect(self, ro, ind, wav):
        dir = spherical(ti.random() * 2 - 1, ti.random())
        pos = dir * self.rad[ind] + self.pos[ind]
        toli = pos - ro
        dis2 = toli.norm_sqr()
        toli = toli.normalized()
        color = tina.rgb_at_wav(self.color[ind], wav)
        wei = color / dis2
        return toli, wei, ti.sqrt(dis2)

    @ti.func
    def background(self, rd, rw):
        if ti.static(hasattr(self, 'skybox')):
            return self.skybox.wav_sample(rd, rw)
        else:
            return 0.0
