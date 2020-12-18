# Tina is a real-time soft renderer based on Taichi for visualizing 3D scenes.
#
# To get started, let's try to make a simple triangle and display it in the GUI.

import taichi as ti
import numpy as np
import tina   # import our renderer

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
# you may disable such face culling policy by using tina.Engine(culling=False)

# to make tina actually display things, we need five things:
#
# 1. Engine - the core implementation of rasterization
engine = tina.Engine()

# 2. Shader - the method to *shade* the object
#
# the shader also wants an field as frame buffer for storing result:
img = ti.Vector.field(3, float, engine.res)
# here we use the `tina.SimpleShader` for simplicity of this tutorial
# basically it shade color by how close the face normal is to view direction
# see docs/lighting.py for advanced shaders with lights and materials
shader = tina.SimpleShader(img)

# 3. GUI - we need to create an window for display (if not offline rendering):
gui = ti.GUI('triangle')
# 4. Control - allows you to control the camera with mouse drags
control = tina.Control(gui)

while gui.running:
    # update the camera transform from the controller
    control.get_camera(engine)

    # clear frame buffer and depth
    img.fill(0)
    engine.clear_depth()

    # specify the mesh vertices
    engine.set_face_verts(verts)
    # render it to image with shader
    engine.render(shader)

    # update the image to GUI
    gui.set_image(img)
    gui.show()
