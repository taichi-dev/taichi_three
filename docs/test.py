import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.Scene()

gui = ti.GUI('test')

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
