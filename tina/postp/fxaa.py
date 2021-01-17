from ..advans import *


@ti.func
def luminance(c):
    return V(0.2989, 0.587, 0.114).dot(c)


# https://catlikecoding.com/unity/tutorials/advanced-rendering/fxaa/
@ti.data_oriented
class FXAA:
    def __init__(self, res):
        self.res = res
        self.lumi = ti.field(float, self.res)
        self.img = ti.Vector.field(3, float, self.res)
        self.abs_thresh = ti.field(float, ())
        self.rel_thresh = ti.field(float, ())
        self.factor = ti.field(float, ())

        @ti.materialize_callback
        def init_params():
            self.abs_thresh[None] = 0.0625
            self.rel_thresh[None] = 0.063
            self.factor[None] = 1

    @ti.kernel
    def apply(self, image: ti.template()):
        for I in ti.grouped(image):
            self.lumi[I] = clamp(luminance(image[I]), 0, 1)
            self.img[I] = image[I]
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
            filt = clamp(filt / c, 0, 1)
            blend = smoothstep(filt, 0, 1)**2 * self.factor[None]
            hori = abs(n + s - 2 * m) * 2
            hori += abs(ne + se - 2 * e)
            hori += abs(nw + sw - 2 * w)
            vert = abs(e + w - 2 * m) * 2
            vert += abs(ne + nw - 2 * n)
            vert += abs(se + sw - 2 * s)
            is_hori = hori >= vert

            plumi = n if is_hori else e
            nlumi = s if is_hori else w
            pgrad = abs(plumi - m)
            ngrad = abs(nlumi - m)

            if pgrad < ngrad:
                blend = -blend
            dir = V(0, 1) if is_hori else V(1, 0)

            image[I] = bilerp(self.img, I + blend * dir)
