import taichi as ti
import taichi_glsl as ts
from .common import *
from .geometry import *
from .camera import *


class RTCamera(Camera):
    @ti.func
    def generate(self, coor):
        eyezoff = 1.0
        orig = ts.vec3(0.0)
        dir = self.cook(ts.vec3(coor, eyezoff))
        return self.trans_pos(orig), self.trans_dir(dir)

    @ti.kernel
    def accumate(self):
        self.scene.curr_camera = ti.static(self)
        self.fb.clear_buffer()
        for model in ti.static(self.scene.models):
            for I in ti.grouped(ti.ndrange(*self.res)):
                orig, dir = self.generate(float(I))
                model.intersect(self, I, orig, dir.normalized())
        self.scene.curr_camera = ti.static(None)