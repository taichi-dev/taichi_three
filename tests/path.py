import taichi as ti
import numpy as np
import taichi_inject
import tina.path

ti.init(ti.cuda)

verts, faces = tina.readobj('assets/monkey.obj', simple=True)
verts = verts[faces]

matr = tina.path.Material()
#geom = tina.path.Particles(matr, np.load('assets/fluid.npy') * 2 - 1, 0.02)
#geom = tina.path.Particles(matr, np.array([[0, 0, 0], [.5, 0, 0]]), 0.2)
geom = tina.path.Triangles(matr, verts)
tree = tina.path.BVHTree(geom=geom)
engine = tina.path.PathEngine(tree)

geom.build(tree)


gui = ti.GUI('BVH', engine.res)
while gui.running and not gui.get_event(gui.ESCAPE, gui.SPACE):
    engine.load_rays()
    for step in range(2):
        engine.step_rays()
    engine.update_image()
    gui.set_image(engine.get_image())
    gui.show()
