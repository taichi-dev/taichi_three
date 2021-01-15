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
    def __init__(self, img, nsamples=64):
        super().__init__(img)
        self.samples = ti.Vector.field(2, float, nsamples)
        self.rotations = ti.Vector.field(2, float, (4, 4))

        @ti.materialize_callback
        @ti.kernel
        def init_ssao():
            for i in self.samples:
                self.samples[i] = self.make_sample()
            for i, j in self.rotations:
                t = ti.tau * ti.random()
                self.rotations[i, j] = V(ti.cos(t), ti.sin(t))

        self.radius = 1.0

    @ti.func
    def make_sample(self):
        t = ti.tau * ti.random()
        r = ti.random()**2
        return V(ti.cos(t), ti.sin(t)) * r

    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        vnormal = mapply_dir(engine.W2V[None], normal).normalized()
        vpos = engine.to_viewspace(pos)

        occ = 0.
        for i in range(self.samples.shape[0]):
            #samp = self.samples[i]
            #rotr = self.rotations[P % self.rotations.shape[0]]
            samp = self.make_sample()
            sample = tangentspace(normal) @ V23(samp, 0.1)
            sample = pos + sample * self.radius
            sample = engine.to_viewspace(sample)
            D = int(engine.to_viewport(sample))
            #D = int(P + 16 * samp)
            if vnormal.xy.dot(D - P) <= 0:
                D = 2 * P - D
            depth = engine.depth[D] / engine.maxdepth
            if depth < vpos.z - eps:
                occ += 1

        ao = occ / self.samples.shape[0]
        self.img[P] = 1 - max(0, ao - 0.16)


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
