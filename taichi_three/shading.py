import taichi as ti
import taichi_glsl as ts
from .transform import *
from .light import Light, AmbientLight
from .skybox import Skybox
from .common import *
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

    def radiance(self, model, pos, indir, texcoor, normal, tangent, bitangent):
        normal = normal.normalized()
        # TODO: we don't support normal maps in path tracing mode for now
        with self.specify_inputs(model=model, pos=pos, texcoor=texcoor, normal=normal, tangent=tangent, bitangent=bitangent, indir=indir) as shader:
            return shader.radiance()

    def pixel_shader(self, model, pos, texcoor, normal, tangent, bitangent):
        # normal has been no longer normalized due to lerp and ndir errors.
        # so here we re-enforce normalization to get slerp.
        normal = normal.normalized()
        with self.specify_inputs(model=model, pos=pos, texcoor=texcoor, normal=normal, tangent=tangent, bitangent=bitangent) as shader:
            return shader.colorize()


@ti.data_oriented
class DeferredMaterial:
    def __init__(self, mtllib, material):
        self.mtllib = mtllib
        self.material = material
        if material not in self.mtllib:
            self.mtllib.append(material)

    @ti.func
    def pixel_shader(self, model, pos, texcoor, normal, tangent, bitangent):
        mid = self.mtllib.index(self.material)
        return dict(mid=mid, position=pos, texcoord=texcoor, normal=normal, tangent=tangent)


@ti.data_oriented
class MaterialVisualizeNormal:
    @ti.func
    def pixel_shader(self, model, pos, texcoor, normal, tangent, bitangent):
        return normal * 0.5 + 0.5


@ti.data_oriented
class MaterialInput:
    def __init__(self, name):
        self.name = name

    def get(self):
        assert hasattr(Material, 'inputs')
        return Material.inputs.get(self.name)


@ti.data_oriented
class Node:
    def __init__(self, **kwargs):
        self.params = dict(self.get_default_params())
        self.params.update(kwargs)
        self._method_level = 0

    @classmethod
    def get_default_params(cls):
        return {}

    def __enter__(self):
        self._method_level += 1
        if self._method_level <= 1:
            for k, v in self.params.items():
                v = v.get()
                setattr(self, k, v)

    def __exit__(self, type, val, tb):
        self._method_level -= 1
        if self._method_level <= 0:
            for k, v in self.params.items():
                if hasattr(self, k):
                    delattr(self, k)

    @staticmethod
    def method(foo):
        from functools import wraps
        @wraps(foo)
        def wrapped(self, *args, **kwargs):
            with self:
                return foo(self, *args, **kwargs)

        return wrapped


class Shading(Node):
    @classmethod
    def get_default_params(cls):
        return dict(
            pos = MaterialInput('pos'),
            normal = MaterialInput('normal'),
            model = MaterialInput('model'),
        )

    @Node.method
    @ti.func
    def radiance(self, pos, indir, normal):
        outdir = indir
        return pos, outdir, ts.vec3(1.0)

    @Node.method
    @ti.func
    def colorize(self):
        pos = self.pos
        normal = self.normal
        res = ts.vec3(0.0)
        viewdir = pos.normalized()
        wpos = (self.model.scene.cameras[-1].L2W @ ts.vec4(pos, 1)).xyz  # TODO: get curr camera?
        if ti.static(self.model.scene.lights):
            for light in ti.static(self.model.scene.lights):
                res += self.render_func(pos, normal, viewdir, light)
        res += self.get_emission()
        return res

    @ti.func
    def render_func(self, pos, normal, viewdir, light):  # TODO: move render_func to Light.render_func?
        if ti.static(isinstance(light, AmbientLight)):
            return light.get_color(pos) * self.get_ambient()
        if ti.static(not isinstance(light, Light)):
            raise NotImplementedError
        lightdir = light.get_dir(pos)
        NoL = ts.dot(normal, lightdir)
        l_out = ts.vec3(0.0)
        if NoL > EPS:
            l_out = light.get_color(pos)
            l_out *= NoL * self.brdf(normal, lightdir, -viewdir)
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
    @classmethod
    def get_default_params(cls):
        return dict(
            pos = MaterialInput('pos'),
            normal = MaterialInput('normal'),
            model = MaterialInput('model'),
            color = Constant(1.0),
            ambient = Constant(1.0),
            specular = Constant(1.0),
            emission = Constant(0.0),
            shineness = Constant(15),
        )

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


# https://zhuanlan.zhihu.com/p/37639418
class CookTorrance(Shading):
    @classmethod
    def get_default_params(cls):
        return dict(
            pos = MaterialInput('pos'),
            normal = MaterialInput('normal'),
            model = MaterialInput('model'),
            color = Constant(1.0),
            ambient = Constant(1.0),
            emission = Constant(0.0),
            roughness = Constant(0.3),
            metallic = Constant(0.0),
            specular = Constant(0.5),
            kd = Constant(1.0),
            ks = Constant(1.0),
            )

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
        f0 = self.metallic * self.color + (1 - self.metallic) * 0.16 * self.specular**2
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


class IdealRT(Shading):
    @classmethod
    def get_default_params(cls):
        return dict(
            pos = MaterialInput('pos'),
            normal = MaterialInput('normal'),
            model = MaterialInput('model'),
            indir = MaterialInput('indir'),
            emission = Constant(0.0),
            diffuse = Constant(1.0),
            specular = Constant(0.0),
            emission_color = Constant(1.0),
            diffuse_color = Constant(1.0),
            specular_color = Constant(1.0),
            )

    @ti.func
    def render_func(self, pos, normal, viewdir, light):  # TODO: move render_func to Light.render_func?
        if ti.static(isinstance(light, Skybox)):
            dir = ts.reflect(viewdir, normal)
            dir = v4trans(self.model.scene.camera.L2W[None], dir, 0)
            return light.sample(dir) * self.specular_color * self.specular
        else:
            return Shading.render_func(self, pos, normal, viewdir, light)

    @Node.method
    @ti.func
    def radiance(self):
        outdir = ts.vec3(0.0)
        clr = ts.vec3(0.0)
        if randomLessThan(self.emission):
            clr = ts.vec3(self.emission_color)
            outdir = ts.vec3(1e-4)
        elif randomLessThan(self.specular):
            clr = ts.vec3(self.specular_color)
            outdir = ts.reflect(self.indir, self.normal)
        elif randomLessThan(self.diffuse):
            clr = ts.vec3(self.diffuse_color)
            outdir = ts.randUnit3D()
            if outdir.dot(self.normal) < 0:
                outdir = -outdir
            #s = ti.random()
            #outdir = ts.vec3(ti.sqrt(1 - s**2) * ts.randUnit2D(), s)
            #outdir = ti.Matrix.cols([self.tangent, self.bitangent, self.normal]) @ outdir
        return self.pos, outdir, clr

    @ti.func
    def brdf(self, normal, lightdir, viewdir):
        NoH = max(0, ts.dot(normal, ts.normalize(lightdir + viewdir)))
        ndf = 5 / 2 * pow(NoH, 12)
        strength = self.diffuse_color * self.diffuse + ndf * self.specular_color * self.specular
        return strength

    @ti.func
    def get_emission(self):
        return self.emission_color * self.emission


class PlaceHolder(Node):
    def __init__(self, hint='placeholder'):
        super().__init__()
        self.hint = hint

    def get(self):
        raise NotImplementedError(hint)


class Constant(Node):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def get(self):
        return self.value


class Uniform(Node):
    def __init__(self, dim, dtype, initial=None):
        super().__init__()
        self.value = create_field(dim, dtype, (), initial)

    @Node.method
    @ti.func
    def get(self):
        return self.value[None]

    def fill(self, value):
        self.value[None] = value


class Texture(Node):
    @classmethod
    def get_default_params(cls):
        return dict(
            texcoor = MaterialInput('texcoor'),
            )

    def __init__(self, texture, scale=None):
        super().__init__()

        if isinstance(texture, str):
            texture = ti.imread(texture)

        # convert UInt8 into Float32 for storage:
        if texture.dtype == np.uint8:
            texture = texture.astype(np.float32) / 255
        elif texture.dtype == np.float64:
            texture = texture.astype(np.float32)

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
                if callable(scale):
                    texture = scale(texture)
                else:
                    texture *= np.array(scale)[None, None, ...]

            # TODO: use create_field for this
            self.texture = ti.Vector.field(3, float, texture.shape[:2])

        @ti.materialize_callback
        def init_texture():
            self.texture.from_numpy(texture)

    @Node.method
    @ti.func
    def get(self):
        return ts.bilerp(self.texture, self.texcoor * ts.vec(*self.texture.shape))

    def fill(self, value):
        self.texture.fill(value)


class NormalMap(Node):
    @classmethod
    def get_default_params(cls):
        return dict(
            texture = PlaceHolder('texture'),
            normal = MaterialInput('normal'),
            tangent = MaterialInput('tangent'),
            bitangent = MaterialInput('bitangent'),
            )

    @Node.method
    @ti.func
    def get(self):
        normal = self.texture * 2 - 1
        normal = ti.Matrix.cols([self.tangent, self.bitangent, self.normal]) @ normal
        return normal.normalized()