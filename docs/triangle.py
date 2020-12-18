# Tina is a real-time soft renderer based on Taichi for visualizing 3D scenes.
#
# To get started, let's try to make a simple triangle and display it in the GUI.

import taichi as ti
import numpy as np
import tina

# Tina use a right-handed coordinate system in world space:
#
# +Z: back.
# +X: right.
# +Y: up.
#
# The camera looks from +Z to target by default.

# make a simple triangle by specifying the vertices
verts = np.array([[
    [-1, -1,  0],  # vertex 1
    [ 1, -1,  0],  # vertex 2
    [ 0,  1,  0],  # vertex 3
    ]])
# also note that face vertices needs to be **counter-clockwise** to be visible
# you may disable such face culling policy by using tina.Scene(culling=False)

# to make tina actually display things, we need at least three things:
#
# 1. Scene - the top structure that manages all resources in the scene
scene = tina.Scene()

# 2. Model - the model to be displayed
#
# here we use `tina.SimpleMesh` which allows use to specify the vertices manually
mesh = tina.SimpleMesh()
# and, don't forget to add the object into the scene so that it gets displayed
scene.add_object(mesh)

# 3. GUI - we also need to create an window for display
gui = ti.GUI('triangle')

while gui.running:
    # update the camera transform from mouse events (will invoke gui.get_events)
    scene.input(gui)

    # set face vertices by feeding a numpy array into it
    mesh.set_face_verts(verts)
    # render image with objects (a triangle in this case) in scene
    scene.render()

    # show the image in GUI
    gui.set_image(scene.img)
    gui.show()
