import taichi as ti
import taichi_glsl as ts
from .light import AmbientLight
from .transform import *
import math


EPS = 1e-4


@ti.data_oriented
class Material:
    def __init__(self, shader):
        self.shader = shader

    def specify_inputs(self, **kwargs):
        Material.inputs = kwargs
        return self
    
    def __enter__(self):
        assert hasattr(Material, 'inputs')
        return self.shader

    def __exit__(self, type, val, tb):
        del Material.inputs


@ti.data_oriented
class Input:
    def __init__(self, name):
        self.name = name

    def get(self):
        assert hasattr(Material, 'inputs')
        return Material.inputs[self.name]


@ti.data_oriented
class Shading:
    use_postp = False

    default_inputs = dict(
        pos=Input('pos'),
        texcoor=Input('texcoor'),
        normal=Input('normal'),
        model=Input('model'),
    )

    def __init__(self, **kwargs):
        self.params = dict(self.get_default_params())
        self.params.update(self.default_inputs)
        self.params.update(kwargs)

    @staticmethod
    def get_default_params():
        raise NotImplementedError

    @ti.func
    def post_process(self, color):
        if ti.static(not self.use_postp):
            return color
        blue = ts.vec3(0.00, 0.01, 0.05)
        orange = ts.vec3(1.19, 1.04, 0.98)
        return ts.mix(blue, orange, ti.sqrt(color))

    @ti.func
    def radiance(self, pos, indir, normal):
        outdir = indir
        return pos, outdir, ts.vec3(1.0)

    @ti.func
    def colorize(self):
        pos = self.pos
        normal = self.normal
        res = ts.vec3(0.0)
        viewdir = pos.normalized()
        wpos = (self.model.scene.cameras[-1].L2W @ ts.vec4(pos, 1)).xyz  # TODO: get curr camera?
        if ti.static(self.model.scene.lights):
            for light in ti.static(self.model.scene.lights):
                strength = light.shadow_occlusion(wpos)
                if strength >= 1e-3:
                    subclr = self.render_func(pos, normal, viewdir, light)
                    res += strength * subclr
        res += self.get_emission()
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
    def get_emission(self):
        return 0

    @ti.func
    def get_ambient(self):
        return 0


class BlinnPhong(Shading):
    color = 1.0
    ambient = 1.0
    specular = 1.0
    emission = 0.0
    shineness = 15

    parameters = ['color', 'ambient', 'specular', 'emission', 'shineness']

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @ti.func
    def get_ambient(self):
        return self.ambient * self.color

    @ti.func
    def get_emission(self):
        return self.emission

    @ti.func
    def brdf(self, normal, lightdir, viewdir):
        NoH = max(0, ts.dot(normal, ts.normalize(lightdir + viewdir)))
        ndf = (self.shineness + 8) / 8 * pow(NoH, self.shineness)
        strength = self.color + ndf * self.specular
        return strength


class StdMtl(Shading):
    Ka = 1.0
    Kd = 1.0
    Ks = 0.0
    Ke = 0.0
    Ns = 1.0

    parameters = ['Ka', 'Kd', 'Ks', 'Ke', 'Ns']

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @ti.func
    def get_ambient(self):
        return self.Ka

    @ti.func
    def get_emission(self):
        return self.Ke

    @ti.func
    def brdf(self, normal, lightdir, viewdir):
        NoH = max(0, ts.dot(normal, ts.normalize(lightdir + viewdir)))
        ndf = (self.Ns + 8) / 8 * pow(NoH, self.Ns)
        strength = self.Ka + ndf * self.Ks
        return strength


class IdealRT(Shading):
    emission = 0.0
    diffuse = 1.0
    specular = 0.0
    emission_color = 1.0
    diffuse_color = 1.0
    specular_color = 1.0
    parameters = ['emission', 'diffuse', 'specular', 'emission_color', 'diffuse_color', 'specular_color']

    @ti.func
    def radiance(self, pos, indir, normal):
        outdir = ts.vec3(0.0)
        clr = ts.vec3(0.0)
        if ti.random() < self.emission:
            clr = ts.vec3(self.emission_color)
        elif ti.random() < self.specular:
            clr = ts.vec3(self.specular_color)
            outdir = ts.reflect(indir, normal)
        elif ti.random() < self.diffuse:
            clr = ts.vec3(self.diffuse_color)
            outdir = ts.randUnit3D()
            if outdir.dot(normal) < 0:
                outdir = -outdir
        return pos, outdir, clr


# https://zhuanlan.zhihu.com/p/37639418
class CookTorrance(Shading):

    @staticmethod
    def get_default_params():
        return dict(
            color = Constant(1.0),
            ambient = Constant(1.0),
            emission = Constant(0.0),
            roughness = Constant(0.3),
            metallic = Constant(0.0),
            specular = Constant(0.04),
            kd = Constant(1.0),
            ks = Constant(1.0),
            )

    def __enter__(self):
        for k, v in self.params.items():
            v = v.get()
            setattr(self, k, v)

    def __exit__(self, type, val, tb):
        for k, v in self.params.items():
            delattr(self, k)

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

    @ti.func
    def get_emission(self):
        return self.emission


@ti.data_oriented
class Constant:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

@ti.data_oriented
class Uniform:
    def __init__(self, dim, dtype, initial=None):
        self.value = create_field(dim, dtype, (), initial)

    @ti.func
    def get(self):
        return self.value[None]

    def fill(self, value):
        self.value[None] = value

@ti.data_oriented
class Texture:
    def __init__(self, texture, scale=None, type='color'):
        # convert UInt8 into Float32 for storage:
        if texture.dtype == np.uint8:
            texture = texture.astype(np.float32) / 255
        elif texture.dtype == np.float64:
            texture = texture.astype(np.float32)

        # normal maps are stored as [-1, 1] for maximizing FP precision:
        if type == 'normal':
            texture = texture * 2 - 1

        if len(texture.shape) == 3 and texture.shape[2] == 1:
            texture = texture.reshape(texture.shape[:2])

        # either RGB or greyscale
        if len(texture.shape) == 2:
            self.texture = ti.field(float, texture.shape)

        else:
            assert len(texture.shape) == 3, texture.shape
            texture = texture[:, :, :3]
            assert texture.shape[2] == 3, texture.shape
            if scale is not None:
                texture *= np.array(scale)[None, None, ...]

            self.texture = ti.Vector.field(3, float, texture.shape[:2])

        @ti.materialize_callback
        def init_texture():
            self.texture.from_numpy(texture)

    @ti.func
    def get(self):
        texcoor = self.params['texcoor'].get()
        return ts.bilerp(self.texture, texcoor * ts.vec(*self.texture.shape))

    def fill(self, value):
        self.texture.fill(value)