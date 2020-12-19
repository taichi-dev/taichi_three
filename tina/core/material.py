from ..common import *
from ..advans import *


@ti.data_oriented
class Node:
    arguments = []
    defaults = []

    def __init__(self, **kwargs):
        for dfl, key in zip(self.defaults, self.arguments):
            value = kwargs.get(key, None)
            if value is None:
                if dfl is None:
                    raise ValueError(f'`{key}` must specified for `{type(self)}`')
                value = dfl

            if isinstance(value, (int, float, ti.Matrix)):
                value = Const(value)
            elif isinstance(value, (list, tuple)):
                value = Const(V(*value))
            elif isinstance(value, str):
                if any(value.endswith(x) for x in ['.png', '.jpg', '.bmp']):
                    value = Texture(value)
                else:
                    value = Input(value)
            setattr(self, key, value)

    def __call__(self, pars):
        raise NotImplementedError(type(self))


class IMaterial(Node):
    def brdf(self, pars, idir, odir):
        raise NotImplementedError(type(self))

    def ambient(self, pars):
        return V(1., 1., 1.)


class Const(Node):
    def __init__(self, value):
        self.value = value

    @ti.func
    def __call__(self, pars):
        return self.value


class Input(Node):
    def __init__(self, name):
        self.name = name

    @ti.func
    def __call__(self, pars):
        return pars[self.name]


class Texture(Node):
    arguments = ['texcoord']
    defaults = ['texcoord']

    def __init__(self, path, **kwargs):
        self.texture = texture_as_field(path)
        super().__init__(**kwargs)

    @ti.func
    def __call__(self, pars):
        maxcoor = V(*self.texture.shape) - 1
        coor = self.texcoord(pars) * maxcoor
        return bilerp(self.texture, coor)


# http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
class CookTorrance(IMaterial):
    arguments = ['normal', 'roughness', 'metallic', 'specular', 'basecolor']
    defaults = ['normal', 0.4, 0.0, 0.5, 'color']

    @ti.func
    def brdf(self, pars, idir, odir):
        EPS = 1e-10
        roughness = self.roughness(pars)
        metallic = self.metallic(pars)
        specular = self.specular(pars)
        basecolor = self.basecolor(pars)
        nrm = self.normal(pars)

        half = (idir + odir).normalized()
        NoH = max(EPS, half.dot(nrm))
        VoH = max(EPS, idir.dot(half))
        NoL = max(EPS, idir.dot(nrm))
        NoV = max(EPS, odir.dot(nrm))
        HoV = min(1 - EPS, max(EPS, half.dot(odir)))

        # Trowbridge-Reitz GGX microfacet distribution
        den = NoH**2 * (roughness**2 - 1) + 1
        ndf = roughness**2 / (ti.pi * den**2)

        # Smith's method with Schlick-GGX
        #k = (roughness + 1)**2 / 8
        #vdf = 1 / ((NoV * (1 - k) + k) * (NoL * (1 - k) + k))

        # GGX partial geometry term
        tan2 = (1 - VoH**2) / VoH**2
        vdf = 1 / (1 + ti.sqrt(1 + roughness**2 * tan2))

        # Fresnel-Schlick approximation
        f0 = metallic * basecolor + (1 - metallic) * 0.16 * specular**2
        #kf = abs((1 - ior) / (1 + ior))**2
        #f0 = kf * basecolor + (1 - kf) * metallic
        ks, kd = f0, (1 - f0)# * (1 - metallic)
        fdf = f0 + (1 - f0) * (1 - HoV)**5

        return kd * basecolor + ks * fdf * vdf * ndf

    def ambient(self, pars):
        return self.basecolor(pars)


class BlinnPhong(IMaterial):
    arguments = ['normal', 'diffuse', 'specular', 'shineness']
    defaults = ['normal', 'color', 0.1, 32.0]

    @ti.func
    def brdf(self, pars, idir, odir):
        diffuse = self.diffuse(pars)
        specular = self.specular(pars)
        shineness = self.shineness(pars)
        nrm = self.normal(pars)

        half = (odir + idir).normalized()
        ks = (shineness + 8) / 8 * pow(max(0, half.dot(nrm)), shineness)
        return diffuse + ks * specular

    def ambient(self, pars):
        return self.diffuse(pars)


class Lambert(IMaterial):
    arguments = ['color']
    defaults = ['color']

    @ti.func
    def brdf(self, pars, idir, odir):
        return self.color(pars)

    def ambient(self, pars):
        return self.color(pars)
