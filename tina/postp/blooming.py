from ..advans import *


@ti.data_oriented
class Blooming:
    def __init__(self, res):
        self.res = tovector(res)
        self.img = ti.field(float, self.res)
        self.out = ti.field(float, self.res)

    @ti.kernel
    def process(self):
        raise NotImplementedError
