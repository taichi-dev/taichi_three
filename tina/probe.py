from .advans import *


@ti.data_oriented
class ProbeShader:
    def __init__(self, res):
        self.res = res
        self.elmid = ti.field(int, res)
        self.texcoord = ti.Vector.field(2, float, res)

    @ti.kernel
    def clear_buffer(self):
        for I in ti.grouped(self.elmid):
            self.elmid[I] = -1
            self.texcoord[I] *= 0

    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.elmid[P] = f
        self.texcoord[P] = texcoord

    @ti.kernel
    def touch(self, callback: ti.template(), mx: float, my: float, rad: float):
        p = V(mx, my) * self.res
        bot, top = ifloor(p - rad), iceil(p + rad)
        for I in ti.grouped(ti.ndrange((bot.x, top.x + 1), (bot.y, top.y + 1))):
            r = (I - p).norm()
            if r > rad:
                continue
            if self.elmid[I] == -1:
                continue
            callback(self, I, r)
