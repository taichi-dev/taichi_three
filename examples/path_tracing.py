import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
obj = t3.readobj('assets/cube.obj', scale=0.8)
model = t3.Model.from_obj(obj)
model.add_uniform('color', 1.0)
model.add_uniform('emission', 0.0)
scene.add_model(model)
light = t3.Model.from_obj(obj)
light.add_uniform('color', 0.0)
light.add_uniform('emission', 1.0)
scene.add_model(light)
camera = t3.RTCamera(res=(256, 256))
scene.add_camera(camera)

light.L2W[None] = t3.translate(0, 1.6, 0) @ t3.scale(0.2)
gui = ti.GUI('Model', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    camera.loadrays()
    camera.steprays()
    camera.steprays()
    camera.applyrays()
    gui.set_image(camera.img)
    gui.show()
