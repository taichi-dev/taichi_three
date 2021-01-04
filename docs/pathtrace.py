import taichi as ti
import numpy as np
import taichi_inject
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True, texturing=True)
material = tina.Phong(color=[0.25, 0.5, 0.5])
mesh = tina.MeshModel('assets/monkey.obj')
scene.add_object(mesh, material)

gui = ti.GUI('pathtrace', scene.res)

scene.update()
while gui.running:
    scene.input(gui)
    scene.render(nsteps=5)
    gui.set_image(scene.img)
    gui.show()
