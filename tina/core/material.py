from ..advans import *


@ti.data_oriented
class Node:
    arguments = []
    defaults = []

    def __init__(self, **kwargs):
        self.params = {}
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
            self.params[key] = value

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(type(self))

    def param(self, key, *args, **kwargs):
        return self.params[key](*args, **kwargs)


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


class Const(Node):
    # noinspection PyMissingConstructor
    def __init__(self, value):
        self.value = value

    @ti.func
    def __call__(self):
        return self.value


class Param(Node):
    # noinspection PyMissingConstructor
    def __init__(self, dtype=float, dim=None, initial=0):
        if dim is not None:
            self.value = ti.Vector.field(dim, dtype, ())
        else:
            self.value = ti.field(dtype, ())

        self.initial = initial
        if initial != 0:
            @ti.materialize_callback
            def init_value():
                self.value[None] = self.initial

    @ti.func
    def __call__(self):
        return self.value[None]

    def make_slider(self, gui, title, min=0, max=1, step=0.01):
        self.slider = gui.slider(title, min, max, step)
        self.slider.value = self.initial

        @gui.post_show
        def post_show(gui):
            self.value[None] = self.slider.value


class Input(Node):
    g_pars = None

    @staticmethod
    def spec_g_pars(pars):
        Input.g_pars = pars

    @staticmethod
    def clear_g_pars():
        Input.g_pars = None

    # noinspection PyMissingConstructor
    def __init__(self, name):
        self.name = name

    @ti.func
    def __call__(self):
        return Input.g_pars[self.name]


class Texture(Node):
    arguments = ['texcoord']
    defaults = ['texcoord']

    def __init__(self, path, **kwargs):
        self.texture = texture_as_field(path)
        super().__init__(**kwargs)

    @ti.func
    def __call__(self):
        maxcoor = V(*self.texture.shape) - 1
        coor = self.param('texcoord') * maxcoor
        return bilerp(self.texture, coor)


# http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
class CookTorrance(IMaterial):
    arguments = ['normal', 'basecolor', 'roughness', 'metallic', 'specular']
    defaults = ['normal', 'color', 0.4, 0.0, 0.5]

    @ti.func
    def brdf(self, nrm, idir, odir):
        EPS = 1e-10
        roughness = self.param('roughness')
        metallic = self.param('metallic')
        specular = self.param('specular')
        basecolor = self.param('basecolor')

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

    def ambient(self):
        return self.param('basecolor')


class BlinnPhong(IMaterial):
    arguments = ['normal', 'diffuse', 'specular', 'shineness']
    defaults = ['normal', 'color', 0.1, 32.0]

    @ti.func
    def brdf(self, nrm, idir, odir):
        diffuse = self.param('diffuse')
        specular = self.param('specular')
        shineness = self.param('shineness')

        half = (odir + idir).normalized()
        ks = (shineness + 8) / 8 * pow(max(0, half.dot(nrm)), shineness)
        return diffuse + ks * specular

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
