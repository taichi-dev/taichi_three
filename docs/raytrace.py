import taichi as ti
import numpy as np
import taichi_inject
import tina

ti.init(ti.gpu)

scene = tina.Scene(smoothing=True, texturing=True, rtx=True, taa=True)
#material = tina.Phong(color=[0.25, 0.5, 0.5])
#mesh = tina.MeshModel('assets/monkey.obj')
material = tina.Phong(specular=1.0)
mesh = tina.MeshModel('assets/sphere.obj')
scene.add_object(mesh, material)

gui = ti.GUI('raytrace', scene.res)

scene.update()
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
