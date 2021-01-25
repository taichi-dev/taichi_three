import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True)#, texturing=True)
scene.engine.skybox = tina.PlainSkybox()

scene.add_object(tina.MeshModel('assets/monkey.obj'), tina.Lambert())
#scene.add_object(tina.MeshModel('assets/cube.obj'), tina.Emission() * [.9, .4, .9])

gui = ti.GUI('rtao', scene.res)

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render()
    gui.set_image(scene.img)
    gui.show()
