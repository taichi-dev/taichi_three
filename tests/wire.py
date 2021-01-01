# In this episode, you'll learn how to render a wireframe model
#
# This tutorial is based on docs/monkey.py, make sure you check that first

import taichi as ti
import numpy as np
import tina

scene = tina.Scene()

# load the monkey using `tina.MeshModel` node (`tina.SimpleMesh` works too):
model = tina.SimpleMesh(npolygon=2)
scene.add_object(model)


gui = ti.GUI('wireframe')

while gui.running:
    scene.input(gui)
    a = gui.frame * 0.03
    verts = np.array([
        [[0, 0, 0], [np.cos(a), np.sin(a), 0]],
    ], dtype=np.float32)
    model.set_face_verts(verts)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
