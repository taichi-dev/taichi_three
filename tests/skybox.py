import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene()
scene.add_object(tina.MeshModel('assets/shadow.obj'))
scene.lighting.skybox = tina.Atomsphere()

gui = ti.GUI('sky', scene.res)

scene.update()
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
