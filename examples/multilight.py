import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
texture = ti.imread("assets/bug.jpg")
model = t3.Model(obj=t3.readobj('assets/bug.obj', scale=0.6), tex=texture)
scene.add_model(model)
camera = t3.Camera(res=(512, 512))
scene.add_camera(camera)
camera.set(pos=[0, 2, 4], target=[0, 2, 0])

light = t3.Light(dir=[0, 0, -1], color=[1.0, 1.0, 1.0])
scene.add_light(light)
light2 = t3.Light(dir=[0, 1, 0], color=[1.0, 1.0, 1.0])
scene.add_light(light2)
gui = ti.GUI('Model', camera.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    # camera.from_mouse(gui)
    model.L2W.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
