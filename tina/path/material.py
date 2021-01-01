from ..advans import *


@ti.data_oriented
class Material:
    def __init__(self):
        pass

    @ti.func
    def brdf(self, idir, odir, nrm):
        return 1. if idir.dot(nrm) <= 0 and odir.dot(nrm) >= 0 else 0.

    @ti.func
    def cdf(self, u, v, su, sv):
        # f(u, v), g(u, v)
        return u, v

    @ti.func
    def pdf(self, u, v, su, sv):
        # df/du dg/dv - df/dv dg/du
        return 1.0

    @ti.func
    def sample(self, idir, nrm):
        u, v = ti.random(), ti.random()
        axes = tangentspace(nrm)
        spec = reflect(idir, nrm)
        su, sv = unspherical(axes.transpose() @ spec)
        odir = axes @ spherical(*self.cdf(u, v, su, sv))
        odir = odir.normalized()
        brdf = self.brdf(idir, odir, nrm)
        return odir, self.pdf(u, v, su, sv) * brdf
