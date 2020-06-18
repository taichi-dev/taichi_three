import taichi as ti
import taichi_glsl as ts
from .render import *


@ti.data_oriented
class SceneBase:
    def __init__(self, res=None):
        self.res = res or (512, 512)
        self.img = ti.Vector(3, ti.f32, self.res)
        self.light_dir = ti.Vector(3, ti.f32, ())
        self.camera = Camera()
        self.opt = Shader()

    def set_light_dir(self, ldir):
        norm = math.sqrt(sum(x**2 for x in ldir))
        ldir = [x / norm for x in ldir]
        self.light_dir[None] = ldir

    @ti.func
    def cook_coor(self, I):
        scale = ti.static(2 / min(*self.img.shape()))
        coor = (I - ts.vec2(*self.img.shape()) / 2) * scale
        return coor

    @ti.func
    def uncook_coor(self, coor):
        coor_xy = ts.shuffle(coor, 0, 1)
        scale = ti.static(min(*self.img.shape()) / 2)
        I = coor_xy * scale + ts.vec2(*self.img.shape()) / 2
        return I

    def do_render(self):
        raise NotImplementedError

    def render(self):
        if not self.camera.is_set:
            self.camera.set()

        self.do_render()


from .scene_rt import *
from .scene_ge import *
