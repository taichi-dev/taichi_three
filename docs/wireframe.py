# In this episode, you'll learn how to render a wireframe model
#
# This tutorial is based on docs/monkey.py, make sure you check that first

import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene()

# load the monkey using `tina.MeshModel` node (`tina.SimpleMesh` works too):
model = tina.MeshModel('assets/monkey.obj')
# convert the mesh to its wireframe using the `tina.MeshToWire` node:
wiremodel = tina.MeshToWire(model)
# load the desired node to be displayed:
scene.add_object(wiremodel)
#scene.add_object(model)

gui = ti.GUI('wireframe')

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
