# Tina is a real-time soft renderer based on Taichi for visualizing 3D scenes.
#
# To get started, let's try to load and display a monkey model in the GUI.

import taichi as ti
import tina

ti.init(ti.gpu)  # use GPU backend for better speed

# to make tina actually display things, we need at least three things:
#
# 1. Scene - the top structure that manages all resources in the scene
scene = tina.Scene()

# 2. Model - the model to be displayed
#
# here we use `tina.MeshModel` which can load models from OBJ format files
model = tina.MeshModel('assets/monkey.obj')
# and, don't forget to add the model into the scene so that it gets displayed
scene.add_object(model)

# 3. GUI - we also need to create an window for display
gui = ti.GUI('monkey')

while gui.running:
    # update the camera transform from mouse events (will invoke gui.get_events)
    scene.input(gui)

    # render scene to image
    scene.render()

    # show the image in GUI
    gui.set_image(scene.img)
    gui.show()
