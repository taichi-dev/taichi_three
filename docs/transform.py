# In this episode, you'll learn how to make use of nodes for transformation
#
# This tutorial is based on docs/monkey.py, make sure you check that first

import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene()

# load the monkey model with the `tina.MeshModel` node:
model = tina.MeshModel('assets/monkey.obj')
# transform the mesh using the `tina.Transform` node:
tmodel = tina.MeshTransform(model)
# load the desired node to be displayed:
scene.add_object(tmodel)
#scene.add_object(model)

gui = ti.GUI('transform')

while gui.running:
    scene.input(gui)

    # create a matrix representing translation along X-axis
    dx = ti.sin(gui.frame * 0.03)
    matrix = tina.translate([dx, 0, 0])
    # set the model transformation matrix for `tina.Transform` node
    tmodel.set_transform(matrix)

    scene.render()
    gui.set_image(scene.img)
    gui.show()
