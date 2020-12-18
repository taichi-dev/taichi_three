# In this episode, you'll learn how to enable smooth shading in Tina
#
# This tutorial is based on docs/monkey.py, make sure you check that first

import taichi as ti
import tina

ti.init(ti.gpu)

# load the torus model (make sure your OBJ file have normals for smooth shading)
obj = tina.readobj('assets/torus.obj')
# get the vertex positions of the model
verts = tina.objverts(obj)
# get the vertex normals of the model, for smooth shading
norms = tina.objnorms(obj)

# enable smooth shading for the engine
engine = tina.Engine(smoothing=True)

img = ti.Vector.field(3, float, engine.res)
shader = tina.SimpleShader(img)

gui = ti.GUI('smooth')
control = tina.Control(gui)

while gui.running:
    control.get_camera(engine)

    img.fill(0)
    engine.clear_depth()

    # specify the mesh vertices
    engine.set_face_verts(verts)
    # specify the mesh normals, for smooth shading
    engine.set_face_norms(norms)
    engine.render(shader)

    gui.set_image(img)
    gui.show()
