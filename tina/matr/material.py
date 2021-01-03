from ..advans import *
from .nodes import *


class IMaterial(Node):
    @ti.func
    def brdf(self, nrm, idir, odir):
        raise NotImplementedError(type(self))

    @ti.func
    def safe_brdf(self, nrm, idir, odir):
        ret = V(0., 0., 0.)
        if nrm.dot(idir) < 0 < nrm.dot(odir):
            ret = self.brdf(nrm, idir, odir)
        return ret

    @ti.func
    def shade(self, idir, odir):
        nrm = self.param('normal')
        return self.brdf(nrm, idir, odir)

    def ambient(self):
        return 1.

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


# http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
# https://blog.csdn.net/cui6864520fei000/article/details/90033863
# https://neil3d.blog.csdn.net/article/details/83783638
class CookTorrance(IMaterial):
    arguments = ['normal', 'basecolor', 'roughness', 'metallic', 'specular']
    defaults = ['normal', 'color', 0.4, 0.0, 0.5]

    @ti.func
    def brdf(self, nrm, idir, odir):  # idir = L, odir = V
        EPS = 1e-10
        roughness = self.param('roughness')
        metallic = self.param('metallic')
        specular = self.param('specular')
        basecolor = self.param('basecolor')

        half = (idir + odir).normalized()
        NoH = max(EPS, half.dot(nrm))
        NoL = max(EPS, idir.dot(nrm))
        NoV = max(EPS, odir.dot(nrm))
        VoH = min(1 - EPS, max(EPS, half.dot(odir)))
        LoH = min(1 - EPS, max(EPS, half.dot(idir)))

        # Trowbridge-Reitz GGX microfacet distribution
        alpha2 = roughness**2
        denom = NoH**2 * (alpha2 - 1) + 1
        ndf = alpha2 / denom**2

        # Smith's method with Schlick-GGX
        k = (roughness + 1)**2 / 8
        vdf = 1 / ((NoV * (1 - k) + k))
        vdf *= 1 / ((NoL * (1 - k) + k))

        # GGX partial geometry term
        #tan2 = (1 - VoH**2) / VoH**2
        #vdf = 1 / (1 + ti.sqrt(1 + roughness**2 * tan2))

        # Fresnel-Schlick approximation
        f0 = metallic * basecolor + (1 - metallic) * 0.16 * specular**2
        #kf = abs((1 - ior) / (1 + ior))**2
        #f0 = kf * basecolor + (1 - kf) * metallic
        ks, kd = f0, (1 - f0)# * (1 - metallic)
        fdf = f0 + (1 - f0) * (1 - VoH)**5

        return kd * basecolor + ks * fdf * vdf * ndf / 4

    def ambient(self):
        return self.param('basecolor')


class Phong(IMaterial):
    arguments = ['normal', 'diffuse', 'specular', 'shineness']
    defaults = ['normal', 'color', 0.1, 32.0]

    @ti.func
    def brdf(self, nrm, idir, odir):
        diffuse = self.param('diffuse')
        specular = self.param('specular')
        shineness = self.param('shineness')

        rdir = reflect(-odir, nrm)
        VoR = max(0, idir.dot(rdir))
        ks = VoR**shineness * (shineness + 2) / 2
        return diffuse + specular * ks

    def ambient(self):
        return self.param('diffuse')


class Lambert(IMaterial):
    arguments = ['normal', 'color']
    defaults = ['normal', 'color']

    @ti.func
    def brdf(self, nrm, idir, odir):
        return self.param('color')

    def ambient(self):
        return self.param('color')


# noinspection PyMissingConstructor
@ti.data_oriented
class VirtualMaterial(IMaterial):
    def __init__(self, materials, mid):
        self.materials = materials
        self.mid = mid

    @ti.func
    def brdf(self, nrm, idir, odir):
        wei = V(0., 0., 0.)
        for i, mat in ti.static(enumerate(self.materials)):
            if i == self.mid:
                wei = mat.brdf(nrm, idir, odir)
        return wei

    @ti.func
    def sample(self, rd, nrm):
        odir, wei = V(0., 0., 0.), V(0., 0., 0.)
        for i, mat in ti.static(enumerate(self.materials)):
            if i == self.mid:
                odir, wei = mat.sample(rd, nrm)
        return odir, wei


@ti.data_oriented
class MaterialTable:
    def __init__(self):
        self.materials = []

    def clear_materials(self):
        self.materials.clear()

    @ti.func
    def get(self, mtlid):
        ti.static_assert(len(self.materials))
        return tina.VirtualMaterial(self.materials, mtlid)

    def add_material(self, matr):
        self.materials.append(matr)
