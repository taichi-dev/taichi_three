import taichi as ti
import numpy as np
import tina

ti.init(ti.cpu)

scene = tina.PTScene(smoothing=True)
scene.engine.skybox = tina.PlainSkybox()

scene.add_object(tina.MeshModel('assets/monkey.obj'), tina.Lambert())

gui = ti.GUI('fpe', scene.res)

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render()
    gui.set_image(scene.img)
    gui.show()
