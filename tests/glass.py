import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True, texturing=True)
scene.engine.skybox = tina.Skybox('assets/skybox.jpg', cubic=True)
scene.add_object(tina.MeshModel('assets/cube.obj'), tina.Glass())

gui = ti.GUI('bdpt', scene.res)

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render(nsteps=12)
    #scene.render_light(nsteps=6)
    gui.set_image(scene.img)
    gui.show()
