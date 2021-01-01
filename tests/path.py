import taichi as ti
import numpy as np
import taichi_inject
import tina.path

ti.init(ti.cpu)

mesh = tina.MeshModel('assets/sphere.obj')
geometry = tina.path.TriangleTracer(smoothing=True, texturing=True)
#geometry.matr = tina.path.CookTorrance(basecolor=tina.Texture('assets/uv.png'), metallic=0.8, roughness=0.3)
lighting = tina.path.Lighting()
tree = tina.path.BVHTree(geometry)
engine = tina.path.PathEngine(tree, lighting)

geometry.set_object(mesh)
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
