import taichi as ti
import taichi_glsl as ts
from .transform import *
import math


@ti.data_oriented
class Shading:
    @ti.func
    def render_func(self, pos, normal, viewdir, light):
        raise NotImplementedError

    @ti.func
    def post_process(self, color):
        if ti.static(1):
            return color
        blue = ts.vec3(0.00, 0.01, 0.05)
        orange = ts.vec3(1.19, 1.04, 0.98)
        return ti.sqrt(ts.mix(blue, orange, color))

    @ti.func
    def colorize(self, pos, normal):
        res = ts.vec3(0.0)
        viewdir = pos.normalized()
        wpos = self.model.scene.cameras[-1].trans_pos(pos)  # TODO: curr_camera
        if ti.static(self.model.scene.lights):
            for light in ti.static(self.model.scene.lights):
                strength = light.shadow_occlusion(wpos)
                if strength != 0:
                    subclr = self.render_func(pos, normal, viewdir, light)
                    res += strength * subclr
        res = self.post_process(res)
        return res

    @ti.func
    def render_func(self, pos, normal, viewdir, light):
        lightdir = light.get_dir(pos)
        costheta = max(0, ts.dot(normal, lightdir))
        l_out = ts.vec3(0.0)
        if costheta > 0:
            l_out = light.get_color(pos)
            l_out *= costheta * self.brdf(normal, -viewdir, lightdir)
        return l_out


class BlinnPhong(Shading):
    shineness = 10
    specular = 0.5
    diffuse = 0.5

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @ti.func
    def brdf(self, normal, lightdir, viewdir):
        NoH = ts.dot(normal, ts.normalize(lightdir + viewdir))
        NoH = pow(max(NoH, 0), self.shineness)
        strength = self.diffuse
        strength += (self.shineness + 8) / 8 * self.specular * NoH
        return strength / math.pi


# References at https://learnopengl.com/PBR/Theory
# Borrowed from https://github.com/victoriacity/taichimd/blob/1dba9dd825cea33f468ed8516b7e2dc6b8995c41/taichimd/graphics.py#L409
# All credits by @victoriacity
class CookTorrance(Shading):
    eps = 1e-4

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
