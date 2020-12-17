import taichi as ti
import numpy as np
import tina
import sys

ti.init(ti.gpu)

obj = tina.readobj(sys.argv[1], scale='auto')
verts = obj['v'][obj['f'][:, :, 0]]
norms = obj['vn'][obj['f'][:, :, 2]]

engine = tina.Engine((1024, 768), maxfaces=len(verts), smoothing=True)
camera = tina.Camera()

img = ti.Vector.field(3, float, engine.res)

shader = tina.SimpleShader(img)

gui = ti.GUI('visualize', engine.res, fast_gui=True)
control = tina.Control(gui)

accum = tina.Accumator(engine.res)

while gui.running:
    engine.randomize_bias(accum.count[None] <= 1)

    if control.get_camera(camera):
        accum.clear()
    engine.set_camera(camera)

    img.fill(0)
    engine.clear_depth()

    engine.set_face_verts(verts)
    engine.set_face_norms(norms)
    engine.render(shader)

    accum.update(img)
    gui.set_image(accum.img)
    gui.show()
