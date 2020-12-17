from ..common import *


@ti.data_oriented
class MagentaShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord):
        self.img[P] = V(1.0, 0.0, 1.0)


@ti.data_oriented
class PositionShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord):
        pos = mapply_pos(engine.L2W[None], pos)
        self.img[P] = pos


@ti.data_oriented
class DepthShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord):
        self.img[P] = engine.depth[P]


@ti.data_oriented
class NormalShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord):
        normal = (engine.NL2W[None] @ normal).normalized()
        self.img[P] = normal * 0.5 + 0.5


@ti.data_oriented
class ChessboardShader:
    def __init__(self, img, size=0.05):
        self.img = img
        self.size = size

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord):
        self.img[P] = lerp((texcoord // self.size).sum() % 2, 0.4, 0.9)


@ti.func
def calc_view_dir(engine, pos):
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
    def shade_color(self, engine, P, f, pos, normal, texcoord):
        pos = mapply_pos(engine.L2W[None], pos)
        view_dir = calc_view_dir(engine, pos)
        self.img[P] = view_dir * 0.5 + 0.5


@ti.data_oriented
class SimpleShader:
    def __init__(self, img):
        self.img = img

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord):
        normal = (engine.NL2W[None] @ normal).normalized()
        pos = mapply_pos(engine.L2W[None], pos)
        view_dir = calc_view_dir(engine, pos)

        self.img[P] = abs(normal.dot(view_dir))


@ti.data_oriented
class Shader:
    def __init__(self, img, lighting, material):
        self.img = img
        self.lighting = lighting
        self.material = material

    @ti.func
    def shade_color(self, engine, P, f, pos, normal, texcoord):
        pos = mapply_pos(engine.L2W[None], pos)
        normal = (engine.NL2W[None] @ normal).normalized()
        view_dir = calc_view_dir(engine, pos)
        pars = {
            'pos': pos,
            'normal': normal,
            'texcoord': texcoord,
        }

        res = V(0.0, 0.0, 0.0)
        res += self.lighting.get_ambient_light_color()
        for l in ti.smart(self.lighting.get_lights_range()):
            light, lcolor = self.lighting.get_light_data(l)
            light_dir = light.xyz - pos * light.w
            cos_i = normal.dot(light_dir)
            if cos_i > 0:
                light_distance = light_dir.norm()
                light_dir /= light_distance
                lcolor /= light_distance**3
                mcolor = self.material.brdf(pars, light_dir, view_dir)
                res += cos_i * lcolor * mcolor

        self.img[P] = res
