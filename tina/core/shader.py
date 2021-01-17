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

    @ti.func
    def blend_color(self, engine, P, p, factor):
        pass


class ConstShader(IShader):
    def __init__(self, img, value):
        super().__init__(img)
        self.value = value

    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.img[P] = self.value


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


class ViewNormalShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        vnormal = mapply_dir(engine.W2V[None], normal).normalized()
        if ti.static(self.img.n == 2):
            self.img[P] = vnormal.xy
        else:
            self.img[P] = vnormal


class TexcoordShader(IShader):
    @ti.func
    def shade_color(self, engine, P, p, f, pos, normal, texcoord, color):
        self.img[P] = texcoord


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
def calc_view_ray(engine, p):
    p = p / engine.res * 2 - 1
    ro = mapply_pos(engine.V2W[None], V23(p, -1.))
    ro1 = mapply_pos(engine.V2W[None], V23(p, 1.))
    rd = (ro1 - ro).normalized()
    return ro, rd


@ti.func
def calc_viewdir(engine, p):
    ro, rd = calc_view_ray(engine, p)
    return -rd


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

    @ti.func
    def blend_color(self, engine, P, p, factor, color):
        self.img[P] = lerp(factor, self.img[P], color)


class ShaderGroup(IShader):
    def __init__(self, shaders=()):
        self.shaders = shaders

    def shade_color(self, *args):
        for shader in self.shaders:
            shader.shade_color(*args)

    def blend_color(self, *args):
        for shader in self.shaders:
            shader.blend_color(*args)


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
