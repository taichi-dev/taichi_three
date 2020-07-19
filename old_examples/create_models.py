import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
cubic = t3.Model(t3.MeshGen().cube([0, 0, 0], [1, 1, 1]))
cylin = t3.Model(t3.MeshGen().cylinder([0, 0, 0], [0, 1, 0], [0.5, 0, 0], [0, 0, 0.5], 20))
scene.add_model(cylin)
scene.add_model(cubic)

scene.set_light_dir([0.4, 1.5, -1.8])
gui = ti.GUI('Creating models', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    cylin.L2W.offset[None] = [0, -0.5, 0]
    cubic.L2W.offset[None] = [-0.5, -0.5, -0.5]
    scene.camera.from_mouse(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()

