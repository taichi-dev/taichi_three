from ..advans import *


@ti.data_oriented
class _PTMaterial:
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
        brdf = self.brdf(nrm, idir, odir)
        return odir, self.pdf(u, v, su, sv) * brdf


class IMaterial(tina.IMaterial, _PTMaterial):
    pass


# http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
class CookTorrance(tina.CookTorrance, _PTMaterial):
    pass


class Lambert(tina.Lambert, _PTMaterial):
    pass
