import taichi as ti
import numpy as np
import taichi_inject
import tina

ti.init(ti.cpu)

scene = tina.PTScene(smoothing=True, texturing=True)

mesh = tina.MeshTransform(tina.MeshModel('assets/monkey.obj'), tina.translate([0, -0.5, 0]))
material = tina.Lambert()

mesh2 = tina.MeshTransform(tina.MeshModel('assets/sphere.obj'), tina.translate([0, +0.5, 0]))
material2 = tina.Lambert(color=tina.Texture('assets/uv.png'))

scene.add_object(mesh, material)
scene.add_object(mesh2, material2)
scene.build()

scene.lighting.set_lights(np.array([
    [0, 1.38457, -1.44325],
], dtype=np.float32))
scene.lighting.set_light_radii(np.array([
    0.2,
], dtype=np.float32))


gui = ti.GUI('BVH', scene.res)
while gui.running and not gui.get_event(gui.ESCAPE, gui.SPACE):
    scene.render(nsteps=3)
    print(gui.frame + 1, 'samples')
    gui.set_image(scene.img)
    gui.show()
