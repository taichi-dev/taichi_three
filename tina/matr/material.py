from ..advans import *
from .nodes import *


class IMaterial(Node):
    @ti.func
    def brdf(self, nrm, idir, odir):
        raise NotImplementedError(type(self))

    @ti.func
    def wav_brdf(self, nrm, idir, odir, wav):
        color = self.brdf(nrm, idir, odir)
        return tina.rgb_at_wav(color, wav)

    @ti.func
    def emission(self):
        return 0.0

    @ti.func
    def estimate_emission(self):
        return 0.0

    @ti.func
    def ambient(self):
        return 1.0

    # cu = f(u, v), cv = g(u, v)
    # pdf = |df/du dg/dv - df/dv dg/du|
    @ti.func
    def sample(self, idir, nrm, sign, rng):
        u, v = rng.random(), rng.random()
        axes = tangentspace(nrm)
        odir = axes @ spherical(u, v)
        odir = odir.normalized()
        brdf = self.brdf(nrm, idir, odir)
        return odir, brdf, 0.2

    @ti.func
    def wav_sample(self, idir, nrm, sign, rng, wav):
        odir, wei = self.sample(idir, nrm, sign, rng)
        return odir, tina.rgb_at_wav(wei, wav)

    @classmethod
    def cook_for_ibl(cls, tab, precision):
        raise NotImplementedError(cls)

    @ti.func
    def sample_ibl(self, ibl, idir, nrm):
        raise NotImplementedError(type(self))

    def __add__(self, other):
        return AddMaterial(self, other)

    def mix(self, other, factor):
        return MixMaterial(self, other, factor)

    def __mul__(self, factor):
        return ScaleMaterial(self, factor)

    def __rmul__(self, factor):
        return ScaleMaterial(self, factor)


@ti.func
def calc_fresnel_factor(metallic, albedo, specular=0.5):
    f0 = metallic * albedo + (1 - metallic) * 0.16 * specular**2
    return f0


class FresnelFactor(Node):
    arguments = ['metallic', 'albedo', 'specular']
    defaults = [0.0, 1.0, 0.5]

    @ti.func
    def __call__(self):
        albedo = self.param('albedo')
        metallic = self.param('metallic')
        specular = self.param('specular')
        return calc_fresnel_factor(metallic, albedo, specular)


class MixMaterial(IMaterial):
    arguments = ['factor']
    defaults = [0.5]

    def __init__(self, mat1, mat2, factor):
        super().__init__(factor=factor)
        self.mat1 = mat1
        self.mat2 = mat2

    @ti.func
    def brdf(self, nrm, idir, odir):
        fac = self.param('factor')
        wei1 = self.mat1.brdf(nrm, idir, odir)
        wei2 = self.mat2.brdf(nrm, idir, odir)
        return (1 - fac) * wei1 + fac * wei2

    @ti.func
    def ambient(self):
        fac = self.param('factor')
        wei1 = self.mat1.ambient()
        wei2 = self.mat2.ambient()
        return (1 - fac) * wei1 + fac * wei2

    @ti.func
    def emission(self):
        fac = self.param('factor')
        wei1 = self.mat1.emission()
        wei2 = self.mat2.emission()
        return (1 - fac) * wei1 + fac * wei2

    @ti.func
    def estimate_emission(self):
        fac = 0.5#self.param('factor')
        wei1 = self.mat1.estimate_emission()
        wei2 = self.mat2.estimate_emission()
        return (1 - fac) * wei1 + fac * wei2

    @ti.func
    def sample(self, idir, nrm, sign, rng):
        fac = self.param('factor')
        odir = V(0., 0., 0.)
        wei = V(0., 0., 0.)
        rough = 0.
        factor = lerp(Vavg(fac), 0.08, 0.92)
        if rng.random() < factor:
            odir, wei, rough = self.mat2.sample(idir, nrm, sign, rng)
            wei *= fac / factor
        else:
            odir, wei, rough = self.mat1.sample(idir, nrm, sign, rng)
            wei *= (1 - fac) / (1 - factor)
        return odir, wei, rough

    @ti.func
    def wav_sample(self, idir, nrm, sign, rng, wav):
        fac = self.param('factor')
        odir = V(0., 0., 0.)
        wei = 0.
        factor = lerp(Vavg(fac), 0.08, 0.92)
        if rng.random() < factor:
            odir, wei = self.mat2.wav_sample(idir, nrm, sign, rng, wav)
            wei *= fac / factor
        else:
            odir, wei = self.mat1.wav_sample(idir, nrm, sign, rng, wav)
            wei *= (1 - fac) / (1 - factor)
        return odir, wei

    @ti.func
    def sample_ibl(self, ibltab, idir, nrm):
        fac = self.param('factor')
        wei1 = self.mat1.sample_ibl(ibltab, idir, nrm)
        wei2 = self.mat2.sample_ibl(ibltab, idir, nrm)
        return (1 - fac) * wei1 + fac * wei2


class ScaleMaterial(IMaterial):
    arguments = ['factor']
    defaults = [1.0]

    def __init__(self, mat, factor):
        super().__init__(factor=factor)
        self.mat = mat

    @ti.func
    def brdf(self, nrm, idir, odir):
        fac = self.param('factor')
        wei = self.mat.brdf(nrm, idir, odir)
        return fac * wei

    @ti.func
    def ambient(self):
        fac = self.param('factor')
        wei = self.mat.ambient()
        return fac * wei

    @ti.func
    def emission(self):
        fac = self.param('factor')
        wei = self.mat.emission()
        return fac * wei

    @ti.func
    def estimate_emission(self):
        fac = 1.0#self.param('factor')
        wei = self.mat.estimate_emission()
        return fac * wei

    @ti.func
    def sample(self, idir, nrm, sign, rng):
        fac = self.param('factor')
        odir, wei, rough = self.mat.sample(idir, nrm, sign, rng)
        return odir, wei * fac, rough

    @ti.func
    def wav_sample(self, idir, nrm, sign, rng, wav):
        fac = self.param('factor')
        odir, wei = self.mat.wav_sample(idir, nrm, sign, rng, wav)
        return odir, wei * fac

    @ti.func
    def sample_ibl(self, ibls: ti.template(), idir, nrm):
        fac = self.param('factor')
        ibl = ti.static(ibls.get(type(self.mat), ibls))
        wei = self.mat.sample_ibl(ibl, idir, nrm)
        return wei * fac


class AddMaterial(IMaterial):
    arguments = []
    defaults = []

    def __init__(self, mat1, mat2):
        super().__init__()
        self.mat1 = mat1
        self.mat2 = mat2

    @ti.func
    def brdf(self, nrm, idir, odir):
        wei1 = self.mat1.brdf(nrm, idir, odir)
        wei2 = self.mat2.brdf(nrm, idir, odir)
        return wei1 + wei2

    @ti.func
    def ambient(self):
        wei1 = self.mat1.ambient()
        wei2 = self.mat2.ambient()
        return wei1 + wei2

    @ti.func
    def emission(self):
        wei1 = self.mat1.emission()
        wei2 = self.mat2.emission()
        return wei1 + wei2

    @ti.func
    def estimate_emission(self):
        wei1 = self.mat1.estimate_emission()
        wei2 = self.mat2.estimate_emission()
        return wei1 + wei2

    @ti.func
    def sample(self, idir, nrm, sign, rng):
        odir = V(0., 0., 0.)
        wei = V(0., 0., 0.)
        rough = 0.
        if rng.random_int() % 2 == 0:
            odir, wei, rough = self.mat1.sample(idir, nrm, sign, rng)
            wei *= 2
        else:
            odir, wei, rough = self.mat2.sample(idir, nrm, sign, rng)
            wei *= 2
        return odir, wei

    @ti.func
    def wav_sample(self, idir, nrm, sign, rng, wav):
        odir = V(0., 0., 0.)
        wei = 0.
        if rng.random_int() % 2 == 0:
            odir, wei = self.mat1.wav_sample(idir, nrm, sign, rng, wav)
            wei *= 2
        else:
            odir, wei = self.mat2.wav_sample(idir, nrm, sign, rng, wav)
            wei *= 2
        return odir, wei


# http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx
# https://blog.csdn.net/cui6864520fei000/article/details/90033863
# https://neil3d.blog.csdn.net/article/details/83783638
class CookTorrance(IMaterial):
    arguments = ['roughness', 'fresnel']
    defaults = [0.4, 1.0]

    def ambient(self):
        return 1.0

    #rough_levels = [0.5, 1.0]
    rough_levels = [0.03, 0.08, 0.18, 0.35, 0.65, 1.0]
    rough_levels = [(a, a - b) for a, b in zip(rough_levels, [0] + rough_levels)]

    @ti.func
    def sample_ibl(self, tab: ti.template(), idir, nrm):
        ibls, lut = ti.static(tab['spec'], tab['lut'])
        # https://zhuanlan.zhihu.com/p/261005894
        roughness = self.param('roughness')
        f0 = self.param('fresnel')

        rdir = reflect(-idir, nrm)
        wei = V(1., 1., 1.)
        for id, rough_info in ti.static(enumerate(self.rough_levels)):
            rough, rough_step = rough_info
            if rough - rough_step <= roughness <= rough:
                wei1 = ibls[id].sample(rdir)
                wei2 = ibls[id + 1].sample(rdir)
                fac2 = (rough - roughness) / rough_step
                wei = lerp(fac2, wei2, wei1)

        EPS = 1e-10
        half = (idir + rdir).normalized()
        VoH = min(1 - EPS, max(EPS, half.dot(rdir)))
        AB = bilerp(lut, V(VoH, roughness))
        wei *= f0 * AB.x + AB.y
        return wei

    @classmethod
    def cook_for_ibl(cls, tab, precision):
        env = tab['env']

        @ti.kernel
        def bake(ibl: ti.template(),
                roughness: ti.template(),
                nsamples: ti.template()):
            # https://zhuanlan.zhihu.com/p/261005894
            for I in ti.grouped(ibl.img):
                dir = ibl.unmapcoor(I)
                res, dem = V(0., 0., 0.), 0.
                alpha2 = max(0, roughness**2)
                for s in range(nsamples):
                    u, v = ti.random(), ti.random()
                    u = ti.sqrt((1 - u) / (1 - u * (1 - alpha2)))
                    odir = tangentspace(dir) @ spherical(u, v)
                    wei = env.sample(odir)
                    res += wei * u
                    dem += u
                ibl.img[I] = res / dem

        lut = texture_as_field('assets/lut.jpg')  # TODO: bake LUT?

        ibls = [env]
        resolution = env.resolution
        for roughness, rough_step in cls.rough_levels:
            resolution = int(resolution / 2**(rough_step * 4))
            ibl = tina.Skybox(resolution)
            ibls.append(ibl)

        @ti.materialize_callback
        def init_ibl():
            nsamples = 8 * precision
            for ibl, (roughness, rough_step) in zip(ibls[1:], cls.rough_levels):
                print(f'[Tina] Baking IBL map ({"x".join(map(str, ibl.shape))} {nsamples} spp) for CookTorrance with roughness {roughness}...')
                bake(ibl, roughness, nsamples)
                nsamples = int(nsamples * 3**(rough_step * 4))
            print('[Tina] Baking IBL map for CookTorrance done')

        tab['spec'] = tuple(ibls)
        tab['lut'] = lut

    @ti.func
    def sub_brdf(self, nrm, idir, odir):  # idir = L, odir = V
        roughness = self.param('roughness')
        f0 = self.param('fresnel')
        EPS = 1e-10

        half = (idir + odir).normalized()
        NoH = max(EPS, half.dot(nrm))
        NoL = max(EPS, idir.dot(nrm))
        NoV = max(EPS, odir.dot(nrm))
        VoH = min(1 - EPS, max(EPS, half.dot(odir)))
        LoH = min(1 - EPS, max(EPS, half.dot(idir)))

        # Trowbridge-Reitz GGX microfacet distribution
        alpha2 = max(0, roughness**2)
        denom = 1 - NoH**2 * (1 - alpha2)
        ndf = alpha2 / denom**2  # D

        # Smith's method with Schlick-GGX
        k = (roughness + 1)**2 / 8
        vdf = 0.5 / ((NoV * k + 1 - k))
        vdf *= 0.5 / ((NoL * k + 1 - k))  # G

        # GGX partial geometry term
        #tan2 = (1 - VoH**2) / VoH**2
        #vdf = 1 / (1 + ti.sqrt(1 + roughness**2 * tan2))

        # Fresnel-Schlick approximation
        #kf = abs((1 - ior) / (1 + ior))**2
        #f0 = kf * basecolor + (1 - kf) * metallic
        fdf = f0 + (1 - f0) * (1 - VoH)**5  # F

        return fdf, vdf, ndf

    @ti.func
    def brdf(self, nrm, idir, odir):
        fdf, vdf, ndf = self.sub_brdf(nrm, idir, odir)
        return fdf * vdf * ndf

    @ti.func
    def sample(self, idir, nrm, sign, rng):
        roughness = self.param('roughness')  # TODO: param duplicate evaluation?
        alpha2 = max(0, roughness**2)
        EPS = 1e-10

        # https://zhuanlan.zhihu.com/p/95865910
        u, v = rng.random(), rng.random()
        u = ti.sqrt((1 - u) / (1 - u * (1 - alpha2)))
        rdir = reflect(-idir, nrm)
        axes = tangentspace(rdir)
        odir = axes @ spherical(u, v)

        fdf, vdf, ndf = self.sub_brdf(nrm, idir, odir)
        if odir.dot(nrm) < 0:
            odir = -odir
            fdf = 0.0  # TODO: fix energy loss on border

        return odir, fdf, roughness


class Lambert(IMaterial):
    arguments = []
    defaults = []

    @ti.func
    def brdf(self, nrm, idir, odir):
        return 1.0

    def ambient(self):
        return 1.0

    @ti.func
    def sample(self, idir, nrm, sign, rng):
        u, v = rng.random(), rng.random()
        axes = tangentspace(nrm)
        odir = axes @ spherical(u, v)
        odir = odir.normalized()
        return odir, 1.0, 1.0

    @classmethod
    def cook_for_ibl(cls, tab, precision):
        env = tab['env']
        ibl = tina.Skybox(env.resolution // 6)
        denoise = tina.Denoise(ibl.shape)
        nsamples = 128 * precision

        @ti.kernel
        def bake():
            # https://zhuanlan.zhihu.com/p/261005894
            for I in ti.grouped(ibl.img):
                dir = ibl.unmapcoor(I)
                res, dem = V(0., 0., 0.), 0.
                for s in range(nsamples):
                    u, v = ti.random(), ti.random()
                    odir = tangentspace(dir) @ spherical(u, v)
                    wei = env.sample(odir)
                    res += wei * u
                    dem += u
                ibl.img[I] = res / dem

        @ti.materialize_callback
        def init_ibl():
            print(f'[Tina] Baking IBL map ({"x".join(map(str, ibl.shape))} {nsamples} spp) for Lambert...')
            bake()
            print('[Tina] Denoising IBL map with KNN for Lambert...')
            denoise.src.copy_from(ibl.img)
            denoise.knn()
            ibl.img.copy_from(denoise.dst)
            print('[Tina] Baking IBL map for Lambert done')

        tab['diff'] = ibl

    @ti.func
    def sample_ibl(self, tab, idir, nrm):
        return tab['diff'].sample(nrm)


class Phong(IMaterial):
    arguments = ['shineness']
    defaults = [32.0]

    @ti.func
    def brdf(self, nrm, idir, odir):
        shineness = self.param('shineness')

        rdir = reflect(-idir, nrm)
        VoR = max(0, odir.dot(rdir))
        return VoR**shineness * (shineness + 2) / 2

    def ambient(self):
        return 1.0

    @ti.func
    def sample(self, idir, nrm, sign, rng):
        # https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
        m = self.param('shineness')
        u, v = rng.random(), rng.random()
        u = u**(1 / (m + 1))
        rdir = reflect(-idir, nrm)
        axes = tangentspace(rdir)
        odir = axes @ spherical(u, v)
        wei = 1.0
        if odir.dot(nrm) < 0:
            odir = -odir
            wei = 0.0
        return odir, wei, 0.1


class Glass(IMaterial):
    arguments = ['ior', 'ior0', 'ior1']
    defaults = [1.52, 1.51, 1.53]

    @ti.func
    def brdf(self, nrm, idir, odir):
        return 0.0

    def ambient(self):
        return 0.0

    @ti.func
    def _sample(self, idir, nrm, sign, rng, ior):
        if sign >= 0:
            ior = 1 / ior

        EPS = 1e-10
        rdir = reflect(-idir, nrm)
        f0 = abs((1 - ior) / (1 + ior))**2
        NoV = min(1 - EPS, max(EPS, nrm.dot(rdir)))
        fdf = f0 + (1 - f0) * (1 - NoV)**5

        wei = fdf
        odir = V(0., 0., 0.)
        factor = lerp(Vavg(fdf), 0.08, 0.92)
        if rng.random() < factor:
            odir = rdir
            wei = fdf / factor
        else:
            has_r, odir = refract(-idir, nrm, ior)
            if has_r == 0:
                odir = rdir
            wei = (1 - fdf) / (1 - factor)
        return odir, wei, 0.0

    @ti.func
    def sample(self, idir, nrm, sign, rng):
        ior = self.param('ior')
        return self._sample(idir, nrm, sign, rng, ior)

    @ti.func
    def wav_sample(self, idir, nrm, sign, rng, rw):
        ior0, ior1 = self.param('ior0'), self.param('ior1')
        ior = lerp(unlerp(rw, 780, 380), ior0, ior1)
        return self._sample(idir, nrm, sign, rng, ior)


class Transparent(IMaterial):
    @ti.func
    def brdf(self, nrm, idir, odir):
        return 0.0

    def ambient(self):
        return 0.0

    @ti.func
    def sample(self, idir, nrm, sign, rng):
        return -idir, 1.0, 0.0

    @ti.func
    def wav_sample(self, idir, nrm, sign, rng, rw):
        return -idir, 1.0


class Mirror(IMaterial):
    arguments = []
    defaults = []

    @ti.func
    def brdf(self, nrm, idir, odir):
        return 0.0

    def ambient(self):
        return 1.0

    @classmethod
    def cook_for_ibl(cls, env, tab, precision):
        pass

    @ti.func
    def sample_ibl(self, tab, idir, nrm):
        odir = reflect(-idir, nrm)
        return tab['env'].sample(odir)

    @ti.func
    def sample(self, idir, nrm, sign, rng):
        odir = reflect(-idir, nrm)
        return odir, 1.0, 0.0


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
    def sample(self, idir, nrm, sign, rng):
        odir, wei, rough = V(0., 0., 0.), V(0., 0., 0.), 0.
        for i, mat in ti.static(enumerate(self.materials)):
            if i == self.mid:
                odir, wei, rough = mat.sample(idir, nrm, sign, rng)
        return odir, wei, rough

    @ti.func
    def wav_brdf(self, nrm, idir, odir, wav):
        wei = 0.
        for i, mat in ti.static(enumerate(self.materials)):
            if i == self.mid:
                wei = mat.wav_brdf(nrm, idir, odir, wav)
        return wei

    @ti.func
    def wav_sample(self, idir, nrm, sign, rng, wav):
        odir, wei = V(0., 0., 0.), 0.
        for i, mat in ti.static(enumerate(self.materials)):
            if i == self.mid:
                odir, wei = mat.wav_sample(idir, nrm, sign, rng, wav)
        return odir, wei

    @ti.func
    def ambient(self):
        wei = V(0., 0., 0.)
        for i, mat in ti.static(enumerate(self.materials)):
            if i == self.mid:
                wei = mat.ambient()
        return wei

    @ti.func
    def emission(self):
        wei = V(0., 0., 0.)
        for i, mat in ti.static(enumerate(self.materials)):
            if i == self.mid:
                wei = mat.emission()
        return wei

    @ti.func
    def estimate_emission(self):
        wei = 0.
        for i, mat in ti.static(enumerate(self.materials)):
            if i == self.mid:
                wei = mat.estimate_emission()
        return wei


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


class Emission(IMaterial):
    arguments = []
    defaults = []

    @ti.func
    def brdf(self, nrm, idir, odir):
        return 0.0

    @ti.func
    def ambient(self):
        return 0.0

    @ti.func
    def emission(self):
        return 1.0

    @ti.func
    def estimate_emission(self):
        return 1.0

    @ti.func
    def sample(self, idir, nrm, sign, rng):
        return idir, 0.0, 0.0


def Classic(color='color', shineness=32, specular=0.4):
    mat_diff = tina.Lambert() * color
    mat_spec = tina.Phong(shineness=shineness)
    material = tina.MixMaterial(mat_diff, mat_spec, specular)
    return material


def Diffuse(color='color'):
    material = tina.Lambert() * color
    return material


def Lamp(color='color'):
    material = tina.Emission() * color
    return material


def PBR(basecolor='color', metallic=0.0, roughness=0.4, specular=0.5):
    mat_diff = tina.Lambert() * basecolor
    f0 = tina.FresnelFactor(metallic=metallic, albedo=basecolor, specular=specular)
    mat_spec = tina.CookTorrance(roughness=roughness, fresnel=f0)
    material = tina.MixMaterial(mat_diff, mat_spec, f0)
    return material
