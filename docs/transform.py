# In this episode, you'll learn how to make use of nodes for transformation.
#
# This tutorial is based on docs/monkey.py, make sure you check that first

import taichi as ti
import tina

ti.init(ti.gpu)

# load the monkey model with the `tina.MeshModel` node:
mesh = tina.MeshModel('assets/monkey.obj')
# transform the mesh using the `tina.Transform` node:
tmesh = tina.Transform(mesh)

engine = tina.Engine()

img = ti.Vector.field(3, float, engine.res)
shader = tina.SimpleShader(img)

gui = ti.GUI('transform')
control = tina.Control(gui)

while gui.running:
    control.get_camera(engine)

    # create a matrix representing translation along X-axis
    dx = ti.sin(gui.frame * 0.03)
    matrix = tina.translate([dx, 0, 0])
    # set the model transformation matrix
    tmesh.set_transform(matrix)

    img.fill(0)
    engine.clear_depth()

    # choose the mesh node to be displayed:
    if gui.is_pressed(gui.SPACE):
        engine.set_mesh(mesh)   # original mesh, not transformed
    else:
        engine.set_mesh(tmesh)  # transformed new mesh
    engine.render(shader)

    gui.set_image(img)
    gui.show()
