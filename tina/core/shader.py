from ..advans import *


@ti.data_oriented
class MagentaShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord, color):
        self.img[P] = V(1.0, 0.0, 1.0)


@ti.data_oriented
class PositionShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord, color):
        self.img[P] = pos


@ti.data_oriented
class DepthShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord, color):
        self.img[P] = engine.depth[P]


@ti.data_oriented
class NormalShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord, color):
        self.img[P] = normal * 0.5 + 0.5


@ti.data_oriented
class TexcoordShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord, color):
        self.img[P] = V23(texcoord, 0.0)


@ti.data_oriented
class ColorShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord, color):
        self.img[P] = color


@ti.data_oriented
class ChessboardShader:
    def __init__(self, img, size=0.05):
        self.img = img
        self.size = size

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord, color):
        self.img[P] = lerp((texcoord // self.size).sum() % 2, 0.4, 0.9)


@ti.func
def calc_viewdir(engine, pos):
    fwd = mapply_dir(engine.V2W[None], V(0., 0., -1.))
    camera, camera_w = mapply(engine.V2W[None], V(0., 0., 0.), 1)
    dir = (camera - pos * camera_w).normalized()
    if dir.dot(fwd) < 0:
        dir = -dir
    return dir


@ti.data_oriented
class ViewdirShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord, color):
        viewdir = calc_viewdir(engine, pos)
        self.img[P] = viewdir * 0.5 + 0.5


@ti.data_oriented
class SimpleShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord, color):
        viewdir = calc_viewdir(engine, pos)

        self.img[P] = abs(normal.dot(viewdir))


@ti.data_oriented
class Shader:
    def __init__(self, img, lighting, material):
        self.img = img
        self.lighting = lighting
        self.material = material

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord, color):
        viewdir = calc_viewdir(engine, pos)
        tina.Input.spec_g_pars({
            'pos': pos,
            'color': color,
            'normal': normal,
            'texcoord': texcoord,
        })

        res = self.lighting.shade_color(self.material, pos, normal, viewdir)
        self.img[P] = res

        tina.Input.clear_g_pars()


@ti.data_oriented
class RTXShader:
    def __init__(self, img, lighting, tree, material):
        self.img = img
        self.lighting = lighting
        self.material = material
        self.tree = tree

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord, color):
        viewdir = calc_viewdir(engine, pos)
        tina.Input.spec_g_pars({
            'pos': pos,
            'color': color,
            'normal': normal,
            'texcoord': texcoord,
        })

        res = self.lighting.shade_color(self.material, self.tree, pos, normal, viewdir)
        self.img[P] = res

        tina.Input.clear_g_pars()


@ti.data_oriented
class BackgroundShader:
    def __init__(self, img, lighting):
        self.img = img
        self.lighting = lighting

    @ti.func
    def shade_background(self, P, dir):
        res = self.lighting.background(dir)
        self.img[P] = res
