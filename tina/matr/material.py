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

    @ti.func
    def ambient(self):
        return 1.0

    # cu = f(u, v), cv = g(u, v)
    # pdf = |df/du dg/dv - df/dv dg/du|
    @ti.func
    def sample(self, idir, nrm, sign):
        u, v = ti.random(), ti.random()
        axes = tangentspace(nrm)
        odir = axes @ spherical(u, v)
        odir = odir.normalized()
        brdf = self.brdf(nrm, idir, odir)
        return odir, brdf


# http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
# https://blog.csdn.net/cui6864520fei000/article/details/90033863
# https://neil3d.blog.csdn.net/article/details/83783638
# TODO: support importance sampling for PBR
class CookTorrance(IMaterial):
    arguments = ['normal', 'basecolor', 'roughness', 'metallic', 'specular']
    defaults = ['normal', 'color', 0.4, 0.0, 0.5]

    @ti.func
    def brdf(self, nrm, idir, odir):  # idir = L, odir = V
        roughness = self.param('roughness')
        metallic = self.param('metallic')
        specular = self.param('specular')
        basecolor = self.param('basecolor')
        EPS = 1e-10

        half = (idir + odir).normalized()
        NoH = max(EPS, half.dot(nrm))
        NoL = max(EPS, idir.dot(nrm))
        NoV = max(EPS, odir.dot(nrm))
        VoH = min(1 - EPS, max(EPS, half.dot(odir)))
        LoH = min(1 - EPS, max(EPS, half.dot(idir)))

        # Trowbridge-Reitz GGX microfacet distribution
        alpha2 = max(eps, roughness**2)
        denom = 1 - NoH**2 * (1 - alpha2)
        ndf = alpha2 / denom**2

        # Smith's method with Schlick-GGX
        k = (roughness + 1)**2 / 8
        #k = roughness**2 / 2
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

    @ti.func
    def sample(self, idir, nrm, sign):
        roughness = self.param('roughness')
        metallic = self.param('metallic')
        specular = self.param('specular')
        basecolor = self.param('basecolor')
        f0 = metallic * basecolor + (1 - metallic) * 0.16 * specular**2
        # https://computergraphics.stackexchange.com/questions/1515/what-is-the-accepted-method-of-converting-shininess-to-roughness-and-vice-versa
        shine = 2 / max(eps * 10, roughness**1.4) - 2
        odir = V(0., 0., 0.)
        ipdf = V(0., 0., 0.)
        factor = lerp(Vavg(f0), eps * 3, 0.9)
        if ti.random() < factor:
            u, v = ti.random(), ti.random()
            u = lerp(u, eps * 314, 1 - eps * 1428.57)
            rdir = reflect(-idir, nrm)
            u = u ** (1 / (shine + 1))
            axes = tangentspace(rdir)
            odir = axes @ spherical(u, v)
            VoR = clamp(odir.dot(rdir), eps, inf)
            phong_brdf = clamp(VoR ** shine * (shine + 1), eps, inf)
            ipdf = f0 / (factor * phong_brdf)
            if odir.dot(nrm) < 0:
                odir = -odir
                ipdf = 0.0  # TODO: fix energy loss for low shineness
        else:
            u, v = ti.random(), ti.random()
            axes = tangentspace(nrm)
            odir = axes @ spherical(u, v)
            ipdf = (1 - f0) / (1 - factor)
        brdf = self.brdf(nrm, idir, odir)
        brdf = clamp(brdf, eps, inf)
        return odir, brdf * ipdf

    def ambient(self):
        return self.param('basecolor')


class Phong(IMaterial):
    arguments = ['normal', 'color', 'specular', 'shineness']
    defaults = ['normal', 'color', 0.2, 32.0]

    @ti.func
    def brdf(self, nrm, idir, odir):
        color = self.param('color')
        specular = self.param('specular')
        diffuse = color * (1 - specular)
        shineness = self.param('shineness')

        rdir = reflect(-idir, nrm)
        VoR = max(0, odir.dot(rdir))
        ks = VoR**shineness * (shineness + 2) / 2
        return diffuse + specular * ks

    def ambient(self):
        return self.param('color')

    @ti.func
    def sample(self, idir, nrm, sign):
        # https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
        m = self.param('shineness')
        color = self.param('color')
        specular = self.param('specular')
        diffuse = color * (1 - specular)
        u, v = ti.random(), ti.random()
        pdf = V(0., 0., 0.)
        odir = V(0., 0., 0.)
        factor = Vavg(specular) / Vavg(diffuse + specular)
        factor = lerp(factor, 0.003, 0.996)
        if ti.random() < factor:
            pdf = specular / factor
            u = u**(1 / (m + 1))
            rdir = reflect(-idir, nrm)
            axes = tangentspace(rdir)
            odir = axes @ spherical(u, v)
            if odir.dot(nrm) < 0:
                odir = -odir
                pdf = 0.0
        else:
            pdf = diffuse / (1 - factor)
            axes = tangentspace(nrm)
            odir = axes @ spherical(u, v)
        return odir, pdf


class Mirror(IMaterial):
    arguments = ['normal', 'color']
    defaults = ['normal', 'color']

    @ti.func
    def brdf(self, nrm, idir, odir):
        return eps

    def ambient(self):
        return 0.0

    @ti.func
    def sample(self, idir, nrm, sign):
        odir = reflect(-idir, nrm)
        return odir, 1.0


class Glass(IMaterial):
    arguments = ['normal', 'color', 'ior']
    defaults = ['normal', 'color', 1.45]

    @ti.func
    def brdf(self, nrm, idir, odir):
        return eps

    def ambient(self):
        return 0.0

    @ti.func
    def sample(self, idir, nrm, sign):
        ior = self.param('ior')
        color = self.param('color')
        if sign >= 0:
            ior = 1 / ior
        has_r, odir = refract(-idir, nrm, ior)
        if has_r == 0:
            odir = reflect(-idir, nrm)
        return odir, color


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
    def sample(self, idir, nrm, sign):
        odir, wei = V(0., 0., 0.), V(0., 0., 0.)
        for i, mat in ti.static(enumerate(self.materials)):
            if i == self.mid:
                odir, wei = mat.sample(idir, nrm, sign)
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
