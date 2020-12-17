# Tina is a real-time soft renderer based on Taichi for visualizing 3D scenes.
#
# To get started, let's try to load a monkey model and display it in the GUI.

import taichi as ti
import tina   # import our renderer

ti.init(ti.gpu)  # use GPU backend for better speed

# `tina.readobj` is an utility function for loading mesh model from OBJ files
# it will also trianglize the mesh for convenience to store in numpy array.
obj = tina.readobj('assets/monkey.obj')
# get the triangle vertex positions of the model
verts = tina.objverts(obj)
print(verts)

# to make tina actually display things, we need five things:
#
# 1. Engine - the core implementation of rasterization
engine = tina.Engine()
# 2. Camera - define the transformations between spaces
camera = tina.Camera()

# 3. Shader - the method to *shade* the object
#
# the shader also wants an field as frame buffer for storing result:
img = ti.Vector.field(3, float, engine.res)
# here we use the `tina.SimpleShader` for simplicity of this tutorial
# basically it shade color by how close the face normal is to view direction
# see examples/lighting.py for advanced shaders with lights and materials
shader = tina.SimpleShader(img)

# 4. GUI - we need to create an window for display (if not offline rendering):
gui = ti.GUI('monkey')
# 5. Control - allows you to control the camera with mouse drags
control = tina.Control(gui)

while gui.running:
    # update the camera from the controller
    control.get_camera(camera)
    # feed the camera into the engine
    engine.set_camera(camera)

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
