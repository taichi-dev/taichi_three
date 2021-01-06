from ..advans import *


@ti.data_oriented
class ImportanceCDF:
    def __init__(self, N_elms=65536):
        self.N_elms = N_elms
        self.cdf = ti.field(float, N_elms)
        self.ncdf = ti.field(int, ())

    def build(self, pdf):
        self.ncdf[None] = len(pdf)

    @ti.func
    def choice(self):
        return ti.random(int) % self.ncdf[None]