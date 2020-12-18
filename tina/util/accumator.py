from ..common import *


@ti.data_oriented
class Accumator:
    def __init__(self, res):
        self.img = ti.Vector.field(3, float, res)
        self.count = ti.field(int, ())

    @ti.kernel
    def clear(self):
        self.count[None] = 0
        for I in ti.grouped(self.img):
            self.img[I] *= 0

    @ti.kernel
    def update(self, src: ti.template()):
        self.count[None] += 1
        inv_count = 1 / self.count[None]
        for I in ti.grouped(self.img):
            color = src[I]
            self.img[I] *= 1 - inv_count
            self.img[I] += color * inv_count
