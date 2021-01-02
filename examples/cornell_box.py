import taichi as ti
import numpy as np
import taichi_inject
import ezprof
import tina

ti.init(ti.cpu)

scene = tina.PTScene(texturing=True)

# TODO: support materials in load_gltf...
mesh = tina.MeshTransform(tina.MeshModel('assets/cornell.obj'), tina.scale(0.5) @ tina.translate([0, -2, 0]))
material = tina.CookTorrance()

scene.add_object(mesh, material)

if isinstance(scene, tina.PTScene):
    scene.lighting.set_lights(np.array([
        [0, 0.92, 0],
    ], dtype=np.float32))
    scene.lighting.set_light_radii(np.array([
        0.078,
    ], dtype=np.float32))
    scene.lighting.set_light_colors(np.array([
        [1.0, 1.0, 1.0],
    ], dtype=np.float32))

if isinstance(scene, tina.PTScene):
    scene.update()

gui = ti.GUI('cornell_box', scene.res)
while gui.running:
    scene.input(gui)
    if isinstance(scene, tina.PTScene):
        with ezprof.scope('render'):
            scene.render(nsteps=5)
        print(gui.frame + 1, 'samples')
    else:
        scene.render()
    gui.set_image(scene.img)
    gui.show()

ezprof.show()