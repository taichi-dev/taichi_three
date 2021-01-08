import taichi as ti
import numpy as np
import taichi_inject
import tina

ti.init(ti.gpu)

scene = tina.Scene(smoothing=True, texturing=True, rtx=True, taa=True)
material = tina.Phong(shineness=64)
mesh = tina.PrimitiveMesh.sphere()
#mesh = tina.MeshModel('assets/monkey.obj')
scene.add_object(mesh, material)

gui = ti.GUI('raytrace', scene.res)

scene.update()
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
