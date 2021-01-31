import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True, texturing=True)
scene.load_gltf('assets/cornell.gltf')
scene.add_object(tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
    tina.translate([0, 3.98, 0]) @ tina.scale(0.1)), tina.Lamp(color=64))

if isinstance(scene, tina.PTScene):
    scene.update()

gui = ti.GUI('cornell_box', scene.res)
scene.init_control(gui, center=(0, 2, 0), radius=5)

while gui.running:
    scene.input(gui)
    if isinstance(scene, tina.PTScene):
        scene.render(nsteps=8)
    else:
        scene.render()
    gui.set_image(scene.img)
    gui.show()

ti.imwrite(scene.img, 'cornell.png')
