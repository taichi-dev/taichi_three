from ..advans import *


@ti.data_oriented
class IShader:
    def __init__(self, img):
        self.img = img

    def clear_buffer(self):
        self.img.fill(0)

    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        raise NotImplementedError


class MagentaShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.img[P] = V(1.0, 0.0, 1.0)


class PositionShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.img[P] = pos


class DepthShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.img[P] = engine.depth[P]


class NormalShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.img[P] = normal


class Normal2DShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        normal = mapply_dir(engine.W2V[None], normal).normalized()
        self.img[P] = normal.xy


class SSAOShader(IShader):
    def __init__(self, res, nsamples=64, thresh=0.02,
            radius=0.2, factor=1.0, blur_rad=2, rot_rad=3):
        self.res = tovector(res)
        self.blur_rad = blur_rad
        self.img = ti.field(float, self.res)
        self.out = ti.field(float, self.res)
        self.samples = ti.Vector.field(3, float, nsamples)
        self.rotations = ti.Vector.field(2, float, (rot_rad, rot_rad, 1))
        self.radius = ti.field(float, ())
        self.thresh = ti.field(float, ())
        self.factor = ti.field(float, ())

        @ti.materialize_callback
        def init_params():
            self.radius[None] = radius
            self.thresh[None] = thresh
            self.factor[None] = factor

        ti.materialize_callback(self.seed_samples)

    @ti.kernel
    def seed_samples(self):
        for i in self.samples:
            self.samples[i] = self.make_sample()
        for I in ti.grouped(self.rotations):
            t = ti.tau * ti.random()
            self.rotations[I] = V(ti.cos(t), ti.sin(t))

    @ti.kernel
    def apply(self, out: ti.template()):
        for i, j in self.img:
            if ti.static(not self.blur_rad):
                out[i, j] *= 1 - self.img[i, j]
            else:
                r, w = 0.0, 0.0
                for k, l in ti.ndrange(self.blur_rad + 1, self.blur_rad + 1):
                    fac = smoothstep(k**2 + l**2, 1.5 * self.blur_rad**2, 0)
                    t = self.img[i - k, j - l]
                    t += self.img[i + k, j + l]
                    t += self.img[i + k, j - l]
                    t += self.img[i - k, j + l]
                    r += t * fac / 4
                    w += fac
                out[i, j] *= 1 - r / w

    @ti.func
    def make_sample(self):
        u, v = ti.random(), ti.random()
        r = lerp(ti.random()**1.5, 0.01, 1.0)
        u = lerp(u, 0.01, 1.0)
        return spherical(u, v) * r

    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        vnormal = mapply_dir(engine.W2V[None], normal).normalized()
        viewdir = calc_viewdir(engine, p)
        vpos = engine.to_viewspace(pos)

        occ = 0.0
        radius = self.radius[None]
        vradius = engine.to_viewspace(pos - radius * viewdir).z - vpos.z
        for i in range(self.samples.shape[0]):
            samp = self.samples[i]
            rot = self.rotations[P % self.rotations.shape[0],
                    i % self.rotations.shape[2]]
            rotmat = ti.Matrix([[rot.x, rot.y], [-rot.x, rot.y]])
            samp.x, samp.y = rotmat @ samp.xy
            #samp = self.make_sample()
            sample = tangentspace(normal) @ samp
            sample = pos + sample * radius
            sample = engine.to_viewspace(sample)
            D = engine.to_viewport(sample)
            #if vnormal.xy.dot(D - P) <= 0:
                #D = 2 * P - D
            if all(0 <= D < engine.res):
                depth = engine.depth[int(D)] / engine.maxdepth
                if depth < sample.z:
                    rc = vradius / (vpos.z - depth)
                    rc = smoothstep(abs(rc), 0, 1)
                    occ += rc

        ao = occ / self.samples.shape[0]
        ao = self.factor[None] * (ao - self.thresh[None])
        self.img[P] = clamp(ao, 0, 1)


class TexcoordShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.img[P] = V23(texcoord, 0.0)


class ColorShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.img[P] = color


class ChessboardShader(IShader):
    def __init__(self, img, size=8):
        super().__init__(img)
        self.size = size

    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.img[P] = lerp((p // self.size).sum() % 2, 0.4, 0.9)


@ti.func
def calc_viewdir(engine, p):
    p = p / engine.res * 2 - 1
    ro = mapply_pos(engine.V2W[None], V23(p, -1.))
    ro1 = mapply_pos(engine.V2W[None], V23(p, 1.))
    dir = (ro - ro1).normalized()
    return dir


@ti.data_oriented
class ViewdirShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        viewdir = calc_viewdir(engine, p)
        self.img[P] = viewdir * 0.5 + 0.5


@ti.data_oriented
class SimpleShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        viewdir = calc_viewdir(engine, p)
        self.img[P] = abs(normal.dot(viewdir))


class Shader(IShader):
    def __init__(self, img, lighting, material):
        super().__init__(img)
        self.lighting = lighting
        self.material = material

    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        viewdir = calc_viewdir(engine, p)
        tina.Input.spec_g_pars({
            'pos': pos,
            'color': color,
            'normal': normal,
            'texcoord': texcoord,
        })

        res = self.lighting.shade_color(self.material, pos, normal, viewdir)
        self.img[P] = res

        tina.Input.clear_g_pars()


class RTXShader(IShader):
    def __init__(self, img, lighting, geom, material):
        super().__init__(img)
        self.lighting = lighting
        self.material = material
        self.geom = geom

    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        viewdir = calc_viewdir(engine, p)
        tina.Input.spec_g_pars({
            'pos': pos,
            'color': color,
            'normal': normal,
            'texcoord': texcoord,
        })

        res = self.lighting.shade_color(self.material, self.geom, pos, normal, viewdir)
        self.img[P] = res

        tina.Input.clear_g_pars()


@ti.data_oriented
class BackgroundShader(IShader):
    def __init__(self, img, lighting):
        super().__init__(img)
        self.lighting = lighting

    @ti.func
    def shade_background(self, P, dir):
        res = self.lighting.background(dir)
        self.img[P] = res
