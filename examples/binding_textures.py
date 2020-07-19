import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model(obj=t3.readobj('assets/cube.obj', scale=0.6),
                 tex=ti.imread('assets/cloth.jpg'))
scene.add_model(model)

scene.set_light_dir([0.4, -1.5, -1.8])
gui = ti.GUI('Model', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    model.L2W.from_mouse(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
