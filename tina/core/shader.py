from ..common import *


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
            'viewdir': viewdir,
        })

        res = V(0.0, 0.0, 0.0)
        res += self.lighting.get_ambient_light_color() * self.material.ambient()
        for l in ti.smart(self.lighting.get_lights_range()):
            light, lcolor = self.lighting.get_light_data(l)
            light_dir = light.xyz - pos * light.w
            light_distance = light_dir.norm()
            light_dir /= light_distance
            cos_i = normal.dot(light_dir)
            if cos_i > 0:
                lcolor /= light_distance**2
                mcolor = self.material.shade(light_dir, viewdir)
                res += cos_i * lcolor * mcolor

        self.img[P] = res

        tina.Input.clear_g_pars()
