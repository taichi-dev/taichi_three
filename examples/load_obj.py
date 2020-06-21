import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
import numpy as np


ti.init(ti.cpu)

scene = t3.Scene()

model = t3.Model(t3.readobj('examples/monkey.obj', scale=0.5))

scene.add_model(model)
scene.set_light_dir([0.4, -1.5, -1.8])


gui = ti.GUI('Loading Model', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    scene.camera.from_mouse(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
