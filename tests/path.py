import taichi as ti
import numpy as np
import tina
import cv2

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True, texturing=True)
material = tina.Phong(color=[0.25, 0.5, 0.5])
mesh = tina.MeshModel('assets/monkey.obj')
scene.add_object(mesh, material)

gui = ti.GUI('pathtrace', scene.res)

@ti.data_oriented
class PostDenoiser:
    def __init__(self, res):
        self.res = res
        self.inp = ti.Vector.field(3, float, res)
        self.out = ti.Vector.field(3, float, res)

    @ti.kernel
    def process(self):
        for i, j in self.inp:
            res = self.inp[i, j]
            res = tina.aces_tonemap(res)
            self.out[i, j] = res

postp = PostDenoiser(scene.res)

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render(nsteps=5)
    scene.engine.get_image(postp.inp)
    postp.process()
    gui.set_image(postp.out)
    gui.show()
