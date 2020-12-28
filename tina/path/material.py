from ..advans import *


@ti.data_oriented
class Material:
    def __init__(self, shineness=10):
        self.shineness = shineness

    @ti.func
    def brdf(self, idir, odir, nrm):
        return 1

    @ti.func
    def f(self, u, v, su, sv):
        # f(u, v), g(u, v)
        return u, v

    @ti.func
    def df(self, u, v, su, sv):
        # df/du dg/dv - df/dv dg/du
        return 1.0

    @ti.func
    def sample(self, idir, nrm):
        spec = reflect(idir, nrm)
        u, v = ti.random(), ti.random()
        axes = tangentspace(nrm)
        su, sv = unspherical(axes.transpose() @ spec)
        odir = axes @ spherical(*self.f(u, v, su, sv))
        odir = odir.normalized()
        brdf = self.brdf(idir, odir, nrm)
        return odir, self.df(u, v, su, sv) * brdf
