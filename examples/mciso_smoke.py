import taichi as ti
import numpy as np
import tina

from tina.util.mciso import MCISO, extend_bounds

ti.init(arch=ti.opengl)


vol = np.load('assets/smoke.npy')
vol = extend_bounds(vol)
mciso = MCISO(vol.shape[0])

engine = tina.Engine((1024, 768), maxfaces=2**20, smoothing=True)

img = ti.Vector.field(3, float, engine.res)
shader = tina.SimpleShader(img)

gui = ti.GUI('mciso_smoke', engine.res, fast_gui=True)
control = tina.Control(gui)
control.center[:] = [0.5, 0.5, 0.5]
control.radius = 1.5

accum = tina.Accumator(engine.res)

mciso.clear()
mciso.m.from_numpy(vol * 4)
mciso.march()

faces, verts, norms = mciso.get_mesh()
verts = verts[faces]
norms = norms[faces]

while gui.running:
    engine.randomize_bias(accum.count[None] <= 1)
    if control.get_camera(engine):
        accum.clear()

    img.fill(0)
    engine.clear_depth()

    engine.set_face_verts(verts)
    engine.set_face_norms(norms)
    engine.render(shader)

    accum.update(img)
    gui.set_image(accum.img)
    gui.show()
