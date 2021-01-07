# In this episode, you'll learn how to render a wireframe model in Tina
#
# This tutorial is based on docs/monkey.py, make sure you check that first

import taichi as ti
import tina

ti.init(ti.cpu)

# you may specify the line width for rendering wireframes:
# taa=True turns on Temporal Anti-Aliasing to make lines smoother
scene = tina.Scene(linewidth=2, taa=True)

# load the monkey using `tina.MeshModel` node (`tina.SimpleMesh` works too):
model = tina.MeshModel('assets/monkey.obj')
# convert the mesh to its wireframe using the `tina.MeshToWire` node:
wiremodel = tina.MeshToWire(model)

# add the wireframe model into scene:
scene.add_object(wiremodel)

# add the original model, with a tiny scale:
model = tina.MeshTransform(model, tina.scale(0.9))
scene.add_object(model)

gui = ti.GUI('wireframe')

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
