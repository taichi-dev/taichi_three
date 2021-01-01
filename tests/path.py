import taichi as ti
import numpy as np
import taichi_inject
import tina.path

ti.init(ti.cpu)

verts, faces = tina.readobj('assets/shadow.obj', simple=True)
verts = verts[faces]

matr = tina.path.Material()
#geometry = tina.path.Particles(matr, np.load('assets/fluid.npy') * 2 - 1, 0.02)
#geometry = tina.path.Particles(matr, np.array([[0, -.5, 0], [0, +.5, 0]], dtype=np.float32), 0.3)
geometry = tina.path.Triangles(matr, verts)
lighting = tina.path.Lighting()
tree = tina.path.BVHTree(geometry)
engine = tina.path.PathEngine(tree, lighting)

geometry.build(tree)
lighting.set_lights(np.array([
    [0, 1.38457, -1.44325],
], dtype=np.float32))
lighting.set_light_radii(np.array([
    0.2,
], dtype=np.float32))


gui = ti.GUI('BVH', engine.res)
while gui.running and not gui.get_event(gui.ESCAPE, gui.SPACE):
    engine.load_rays()
    for step in range(3):
        engine.step_rays()
    engine.update_image()
    print(gui.frame + 1, 'samples')
    gui.set_image(engine.get_image())
    gui.show()
