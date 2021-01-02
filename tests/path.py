import taichi as ti
import numpy as np
import taichi_inject
import tina.path

ti.init(ti.cpu)

tracer = tina.path.TriangleTracer(smoothing=True, texturing=True)
mtltab = tina.path.MaterialTable()

lighting = tina.path.Lighting()
tree = tina.path.BVHTree(tracer)
engine = tina.path.PathEngine(tree, lighting, mtltab)

mesh = tina.MeshTransform(tina.MeshModel('assets/monkey.obj'), tina.translate([0, -0.5, 0]))
material = tina.path.Lambert()

mesh2 = tina.MeshTransform(tina.MeshModel('assets/sphere.obj'), tina.translate([0, +0.5, 0]))
material2 = tina.path.Lambert(color=tina.Texture('assets/uv.png'))

mtltab.add_material(material)
tracer.add_object(mesh, 0)
mtltab.add_material(material2)
tracer.add_object(mesh2, 1)
tracer.build(tree)

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
