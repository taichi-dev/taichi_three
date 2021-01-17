from ..advans import *


@ti.func
def luminance(c):
    return V(0.2989, 0.587, 0.114).dot(c)


# https://catlikecoding.com/unity/tutorials/advanced-rendering/fxaa/
@ti.data_oriented
class FXAA:
    def __init__(self, res):
        self.res = res
        self.abs_thresh = ti.field(float, ())
        self.rel_thresh = ti.field(float, ())
        self.lumi = ti.field(float, self.res)

        @ti.materialize_callback
        def init_params():
            self.abs_thresh[None] = 0.0625
            self.rel_thresh[None] = 0.166

    @ti.kernel
    def apply(self, image: ti.template()):
        for I in ti.grouped(image):
            self.lumi[I] = clamp(luminance(image[I]), 0, 1)
            image[I] *= 0
        for I in ti.grouped(image):
            m = self.lumi[I]
            n = self.lumi[I + V(0, 1)]
            e = self.lumi[I + V(1, 0)]
            s = self.lumi[I + V(0, -1)]
            w = self.lumi[I + V(-1, 0)]
            ne = self.lumi[I + V(1, 1)]
            nw = self.lumi[I + V(-1, 1)]
            se = self.lumi[I + V(1, -1)]
            sw = self.lumi[I + V(-1, -1)]
            hi = max(m, n, e, s, w)
            lo = min(m, n, e, s, w)
            c = hi - lo
            if c < self.abs_thresh[None] or c < self.rel_thresh[None] * hi:
                continue
            filt = 2 * (n + e + s + w)
            filt += ne + nw + se + sw
            filt = abs(filt / 12 - m)
            image[I] = filt
