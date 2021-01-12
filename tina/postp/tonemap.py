from .advans import *


@ti.data_oriented
class ToneMapping:
    def __init__(self, inp, res):
        self.inp = inp
        self.out = ti.Vector.field(3, float, res)

    @ti.kernel
    def process(self):
        for I in ti.grouped(self.inp):
            self.out[I] = aces_tonemap(self.inp[I])
