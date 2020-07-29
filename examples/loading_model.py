import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
texture = ti.imread("assets/cloth.jpg")
model = t3.Model(obj=t3.readobj('assets/torus.obj', scale=1))
scene.add_model(model)
camera = t3.Camera()
scene.add_camera(camera)

light = t3.Light([0.4, -1.5, 1.8])
scene.add_light(light)
gui = ti.GUI('Model', camera.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    #model.L2W.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
