import taichi as ti
import numpy as np
import taichi_inject
import tina.path

ti.init(ti.cuda)


matr = tina.path.Material()
#pars = tip.Particles(matr, np.load('assets/fluid.npy') * 2 - 1, 0.02)
pars = tina.path.Particles(matr,
        np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], np.float32), 0.2)
tree = tina.path.BVHTree(geom=pars)
camera = tina.path.Camera(scene=tree)

pars.build(tree)


gui = ti.GUI('BVH', tuple(camera.res.entries))
while gui.running and not gui.get_event(gui.ESCAPE, gui.SPACE):
    camera.load_rays()
    for step in range(6):
        camera.step_rays()
    camera.update_image()
    gui.set_image(camera.get_image())
    gui.show()
