from ..advans import *


@ti.data_oriented
class ToneMapping:
    def __init__(self, res):
        self.res = res

    @ti.kernel
    def apply(self, image: ti.template()):
        for I in ti.grouped(image):
            image[I] = aces_tonemap(image[I])
