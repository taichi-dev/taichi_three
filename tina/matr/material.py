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

    @classmethod
    def cook_for_ibl(cls, env, precision):
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
    def sample(self, idir, nrm, sign):
        fac = self.param('factor')
        odir = V(0., 0., 0.)
        wei = V(0., 0., 0.)
        factor = lerp(Vavg(fac), eps * 3, 1 - eps * 3)
        if ti.random() < factor:
            odir, wei = self.mat2.sample(idir, nrm, sign)
            wei *= fac / factor
        else:
            odir, wei = self.mat1.sample(idir, nrm, sign)
            wei *= (1 - fac) / (1 - factor)
        return odir, wei

    @ti.func
    def sample_ibl(self, ibls: ti.template(), idir, nrm):
        fac = self.param('factor')
        ibl1 = ti.static(ibls.get(type(self.mat1), ibls))
        ibl2 = ti.static(ibls.get(type(self.mat2), ibls))
        wei1 = self.mat1.sample_ibl(ibl1, idir, nrm)
        wei2 = self.mat2.sample_ibl(ibl2, idir, nrm)
        return (1 - fac) * wei1 + fac * wei2  # XXX: avoid ks duplication for LUT


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
    def sample(self, idir, nrm, sign):
        fac = self.param('factor')
        odir, wei = self.mat.sample(idir, nrm, sign)
        wei *= fac
        return odir, wei

    @ti.func
    def sample_ibl(self, ibls: ti.template(), idir, nrm):
        fac = self.param('factor')
        ibl = ti.static(ibls.get(type(self.mat), ibls))
        wei = self.mat.sample_ibl(ibl, idir, nrm)
        return fac * wei


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
        fac = self.param('factor')
        wei1 = self.mat1.ambient()
        wei2 = self.mat2.ambient()
        return wei1 + wei2

    @ti.func
    def sample(self, idir, nrm, sign):
        odir = V(0., 0., 0.)
        wei = V(0., 0., 0.)
        if ti.random(int) % 2 == 0:
            odir, wei = self.mat1.sample(idir, nrm, sign)
            wei *= 2
        else:
            odir, wei = self.mat2.sample(idir, nrm, sign)
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
    def sample_ibl(self, ibl_info: ti.template(), idir, nrm):
        ibls, lut = ti.static(ibl_info)
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
    def cook_for_ibl(cls, env, precision):
        @ti.kernel
        def bake(ibl: ti.template(),
                roughness: ti.template(),
                nsamples: ti.template()):
            # https://zhuanlan.zhihu.com/p/261005894
            for I in ti.grouped(ibl.img):
                dir = ibl.unmapcoor(I)
                res = V(0., 0., 0.)
                alpha2 = max(0, roughness**2)
                for s in range(nsamples):
                    u, v = ti.random(), ti.random()
                    u = ti.sqrt((1 - u) / (1 - u * (1 - alpha2)))
                    odir = tangentspace(dir) @ spherical(u, v)
                    wei = env.sample(odir)
                    res += wei * u
                ibl.img[I] = res / nsamples

        lut = texture_as_field('assets/lut.jpg')

        ibls = [env]
        resolution = env.resolution
        for roughness, rough_step in cls.rough_levels:
            resolution = int(resolution / 2**(rough_step * 4))
            ibl = tina.Skybox(resolution)
            ibls.append(ibl)

        @ti.materialize_callback
        def init_ibl():
            nsamples = 4 * precision
            for ibl, (roughness, rough_step) in zip(ibls[1:], cls.rough_levels):
                print(f'[Tina] Baking IBL map ({"x".join(map(str, ibl.shape))} {nsamples} spp) for CookTorrance with roughness {roughness}...')
                bake(ibl, roughness, nsamples)
                nsamples = int(nsamples * 3**(rough_step * 4))
            print('[Tina] Baking IBL map for CookTorrance done')

        return tuple(ibls), lut

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
    def sample(self, idir, nrm, sign):
        roughness = self.param('roughness')  # TODO: param duplicate evaluation?
        alpha2 = max(0, roughness**2)
        EPS = 1e-10

        # https://zhuanlan.zhihu.com/p/95865910
        u, v = ti.random(), ti.random()
        u = ti.sqrt((1 - u) / (1 - u * (1 - alpha2)))
        rdir = reflect(-idir, nrm)
        axes = tangentspace(rdir)
        odir = axes @ spherical(u, v)

        pdf = 1.0
        if odir.dot(nrm) < 0:
            odir = -odir
            pdf = 0.0  # TODO: fix energy loss on border
        else:
            fdf, vdf, ndf = self.sub_brdf(nrm, idir, odir)
            pdf = fdf

        return odir, pdf


class Lambert(IMaterial):
    arguments = []
    defaults = []

    @ti.func
    def brdf(self, nrm, idir, odir):
        return 1.0

    def ambient(self):
        return 1.0

    @ti.func
    def sample(self, idir, nrm, sign):
        u, v = ti.random(), ti.random()
        axes = tangentspace(nrm)
        odir = axes @ spherical(u, v)
        odir = odir.normalized()
        return odir, 1.0

    @classmethod
    def cook_for_ibl(cls, env, precision):
        ibl = tina.Skybox(env.resolution // 6)
        tmp = ti.Vector.field(3, float, ibl.shape)
        nsamples = 256 * precision

        @ti.kernel
        def bake():
            # https://zhuanlan.zhihu.com/p/261005894
            for I in ti.grouped(ibl.img):
                dir = ibl.unmapcoor(I)
                res = V(0., 0., 0.)
                for s in range(nsamples):
                    u, v = ti.random(), ti.random()
                    odir = tangentspace(dir) @ spherical(u, v)
                    wei = env.sample(odir)
                    res += wei * u
                tmp[I] = res / nsamples
            for I in ti.grouped(ibl.img):
                res = tmp[I]
                if not any(I == 0 or I == V(*ibl.shape) - 1):
                    res *= 4
                    for i in ti.static(range(2)):
                        res += tmp[I + U2(i)] + tmp[I - U2(i)]
                    res /= 8
                ibl.img[I] = res

        @ti.materialize_callback
        def init_ibl():
            print(f'[Tina] Baking IBL map ({"x".join(map(str, ibl.shape))} {nsamples} spp) for Lambert...')
            bake()
            print('[Tina] Baking IBL map for Lambert done')

        return ibl

    @ti.func
    def sample_ibl(self, ibl, idir, nrm):
        return ibl.sample(nrm)


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
    def sample(self, idir, nrm, sign):
        # https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
        m = self.param('shineness')
        u, v = ti.random(), ti.random()
        u = u**(1 / (m + 1))
        rdir = reflect(-idir, nrm)
        axes = tangentspace(rdir)
        odir = axes @ spherical(u, v)
        wei = 1.0
        if odir.dot(nrm) < 0:
            odir = -odir
            wei = 0.0
        return odir, wei


class Glass(IMaterial):
    arguments = ['ior']
    defaults = [1.45]

    @ti.func
    def brdf(self, nrm, idir, odir):
        return 0.0

    def ambient(self):
        return 0.0

    @ti.func
    def sample(self, idir, nrm, sign):
        ior = self.param('ior')
        if sign >= 0:
            ior = 1 / ior

        EPS = 1e-10
        rdir = reflect(-idir, nrm)
        f0 = abs((1 - ior) / (1 + ior))**2
        NoV = min(1 - EPS, max(EPS, nrm.dot(rdir)))
        fdf = f0 + (1 - f0) * (1 - NoV)**5

        wei = 1.0
        odir = V(0., 0., 0.)
        factor = lerp(fdf, 0.08, 0.92)
        if ti.random() < factor:
            odir = rdir
            wei *= fdf / factor
        else:
            has_r, odir = refract(-idir, nrm, ior)
            if has_r == 0:
                odir = rdir
            wei *= (1 - fdf) / (1 - factor)
        return odir, wei


class Mirror(IMaterial):
    arguments = []
    defaults = []

    @ti.func
    def brdf(self, nrm, idir, odir):
        return 0.0

    def ambient(self):
        return 0.0

    @classmethod
    def cook_for_ibl(cls, env, precision):
        return env

    @ti.func
    def sample_ibl(self, ibl, idir, nrm):
        odir = reflect(-idir, nrm)
        return ibl.sample(odir)

    @ti.func
    def sample(self, idir, nrm, sign):
        odir = reflect(-idir, nrm)
        return odir, 1.0


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


def Classic(color=1.0, shineness=32, specular=0.4):
    mat_diff = tina.Lambert() * color
    mat_spec = tina.Phong(shineness=shineness)
    material = tina.MixMaterial(mat_diff, mat_spec, specular)
    return material


def Diffuse(color=1.0):
    material = tina.Lambert() * color
    return material


def PBR(basecolor=1.0, metallic=0.0, roughness=0.4, specular=0.5):
    mat_diff = tina.Lambert() * basecolor
    f0 = tina.FresnelFactor(metallic=metallic, albedo=basecolor, specular=specular)
    mat_spec = tina.CookTorrance(roughness=roughness, fresnel=f0)
    material = tina.MixMaterial(mat_diff, mat_spec, f0)
    return material
