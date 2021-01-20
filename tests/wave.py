import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True)
#scene.lighting.skybox = tina.Atomsphere()
scene.lighting.skybox = tina.Skybox('assets/grass.jpg')

scene.add_object(tina.MeshModel('assets/sphere.obj'), tina.Glass())
scene.add_object(tina.MeshTransform(tina.MeshModel('assets/cube.obj'),
        tina.translate([0, -3, 0]) @ tina.scale(2)), tina.Lambert())

gui = ti.GUI('wave', scene.res)

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render(nsteps=6)
    gui.set_image(scene.img)
    gui.show()
