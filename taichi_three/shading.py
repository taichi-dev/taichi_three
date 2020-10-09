import taichi as ti
import taichi_glsl as ts
from .light import AmbientLight
from .transform import *
import math


EPS = 1e-4


@ti.data_oriented
class Shading:
    use_postp = False

    @ti.func
    def post_process(self, color):
        if ti.static(not self.use_postp):
            return color
        blue = ts.vec3(0.00, 0.01, 0.05)
        orange = ts.vec3(1.19, 1.04, 0.98)
        return ts.mix(blue, orange, ti.sqrt(color))

    @ti.func
    def colorize(self, pos, normal):
        res = ts.vec3(0.0)
        viewdir = pos.normalized()
        wpos = (self.model.scene.cameras[-1].L2W @ ts.vec4(pos, 1)).xyz  # TODO: get curr camera?
        if ti.static(self.model.scene.lights):
            for light in ti.static(self.model.scene.lights):
                strength = light.shadow_occlusion(wpos)
                if strength >= 1e-3:
                    subclr = self.render_func(pos, normal, viewdir, light)
                    res += strength * subclr
        res = self.post_process(res)
        return res

    @ti.func
    def render_func(self, pos, normal, viewdir, light):  # TODO: move render_func to Light.render_func?
        if ti.static(isinstance(light, AmbientLight)):
            return light.get_color(pos) * self.get_ambient()
        lightdir = light.get_dir(pos)
        NoL = ts.dot(normal, lightdir)
        l_out = ts.vec3(0.0)
        if NoL > EPS:
            l_out = light.get_color(pos)
            l_out *= NoL * self.brdf(normal, -viewdir, lightdir)
        return l_out

    def brdf(self, normal, lightdir, viewdir):
        raise NotImplementedError

    @ti.func
    def get_ambient(self):
        raise NotImplementedError


class BlinnPhong(Shading):
    color = 1.0
    ambient = 1.0
    specular = 1.0
    shineness = 15

    parameters = ['color', 'ambient', 'specular', 'shineness']

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @ti.func
    def get_ambient(self):
        return self.ambient * self.color

    @ti.func
    def brdf(self, normal, lightdir, viewdir):
        NoH = max(0, ts.dot(normal, ts.normalize(lightdir + viewdir)))
        ndf = (self.shineness + 8) / 8 * pow(NoH, self.shineness)
        strength = self.color + ndf * self.specular
        return strength


class StdMtl(Shading):
    Ns = 1.0
    Ka = 1.0
    Kd = 1.0
    Ks = 0.0
    Ke = 0.0

    parameters = ['Ka', 'Kd', 'Ks', 'Ke', 'Ns']

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @ti.func
    def get_ambient(self):
        return self.Ka

    @ti.func
    def brdf(self, normal, lightdir, viewdir):
        NoH = max(0, ts.dot(normal, ts.normalize(lightdir + viewdir)))
        ndf = (self.Ns + 8) / 8 * pow(NoH, self.Ns)
        strength = self.Ka + ndf * self.Ks
        return strength


# https://zhuanlan.zhihu.com/p/37639418
class CookTorrance(Shading):
    color = 1.0
    ambient = 1.0
    roughness = 0.3
    metallic = 0.0
    specular = 0.04
    kd = 1.0
    ks = 1.0

    parameters = ['color', 'ambient', 'roughness', 'metallic']

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @ti.func
    def ischlick(self, cost):
        k = (self.roughness + 1)**2 / 8
        return k + (1 - k) * cost

    @ti.func
    def fresnel(self, f0, HoV):
        return f0 + (1 - f0) * (1 - HoV)**5

    @ti.func
    def brdf(self, normal, lightdir, viewdir):
        halfway = ts.normalize(lightdir + viewdir)
        NoH = max(EPS, ts.dot(normal, halfway))
        NoL = max(EPS, ts.dot(normal, lightdir))
        NoV = max(EPS, ts.dot(normal, viewdir))
        HoV = min(1 - EPS, max(EPS, ts.dot(halfway, viewdir)))
        ndf = self.roughness**2 / (NoH**2 * (self.roughness**2 - 1) + 1)**2
        vdf = 0.25 / (self.ischlick(NoL) * self.ischlick(NoV))
        f0 = self.metallic * self.color + (1 - self.metallic) * self.specular
        ks, kd = self.ks * f0, self.kd * (1 - f0) * (1 - self.metallic)
        fdf = self.fresnel(f0, NoV)
        strength = kd * self.color + ks * fdf * vdf * ndf / math.pi
        return strength

    @ti.func
    def get_ambient(self):
        return self.ambient * self.color


# References at https://learnopengl.com/PBR/Theory
# Borrowed from https://github.com/victoriacity/taichimd/blob/1dba9dd825cea33f468ed8516b7e2dc6b8995c41/taichimd/graphics.py#L409
# All credits by @victoriacity
class VictoriaCookTorrance(Shading):
    eps = EPS

    specular = 0.6
    kd = 1.6
    ks = 2.0
    roughness = 0.6
    metallic = 0.0

    '''
    Cook-Torrance BRDF with an Lambertian factor.
    Lo(p, w0)=\int f_c*Li(p, wi)(n.wi)dwi where
    f_c = k_d*f_lambert * k_s*f_cook-torrance
    For finite point lights, the integration is evaluated as a
    discrete sum.
    '''

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    '''
    Calculates the Cook-Torrance BRDF as
    f_lambert = color / pi
    k_s * f_specular = D * F * G / (4 * (wo.n) * (wi.n))
    '''

    @ti.func
    def brdf(self, normal, viewdir, lightdir, color):
        halfway = ts.normalize(viewdir + lightdir)
        ndotv = max(ti.dot(viewdir, normal), self.eps)
        ndotl = max(ti.dot(lightdir, normal), self.eps)
        diffuse = self.kd * color / math.pi
        specular = self.microfacet(normal, halfway)\
                    * self.frensel(viewdir, halfway, color)\
                    * self.geometry(ndotv, ndotl)
        specular /= 4 * ndotv * ndotl
        return diffuse + specular

    '''
    Trowbridge-Reitz GGX microfacet distribution
    '''
    @ti.func
    def microfacet(self, normal, halfway):
        alpha = self.roughness
        ndoth = ts.dot(normal, halfway)
        ggx = alpha**2 / math.pi
        ggx /= (ndoth**2 * (alpha**2 - 1.0) + 1.0)**2
        return ggx

    '''
    Fresnel-Schlick approximation
    '''
    @ti.func
    def frensel(self, view, halfway, color):
        f0 = ts.mix(self.specular, color, self.metallic)
        hdotv = ts.clamp(ts.dot(halfway, view), 0, 1)
        return (f0 + (1 - f0) * (1 - hdotv)**5) * self.ks

    '''
    Smith's method with Schlick-GGX
    '''
    @ti.func
    def geometry(self, ndotv, ndotl):
        k = (self.roughness + 1)**2 / 8
        geom = ndotv * ndotl \
            / (ndotv * (1 - k) + k) / (ndotl * (1 - k) + k)
        return max(0, geom)


    '''
    Compared with the basic lambertian-phong shading,
    the rendering function also takes the surface color as parameter.
    Also note that the viewdir points from the camera to the object
    so it needs to be inverted when calculating BRDF.
    '''
    @ti.func
    def render_func(self, pos, normal, viewdir, light, color):
        lightdir = light.get_dir(pos)
        costheta = max(0, ts.dot(normal, lightdir))
        l_out = ts.vec3(0.0)
        if costheta > 0:
            l_out = self.brdf(normal, -viewdir, lightdir, color) \
                 * costheta * light.get_color(pos)
        return l_out
