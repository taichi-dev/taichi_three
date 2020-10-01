import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
texture = ti.imread("assets/cloth.jpg")
model = t3.Model.from_obj(t3.readobj('assets/monkey.obj', scale=0.8), texture)
scene.add_model(model)
camera = t3.Camera(res=(256, 256), pos=[0, 0, 2.5], target=[0, 0, 0], up=[0, 1, 0])
scene.add_camera(camera)

camera2 = t3.Camera(pos=[0, 1, -2], target=[0, 1, 0], up=[0, 1, 0])
scene.add_camera(camera2)

light = t3.Light([0.4, -1.5, -0.8])
scene.add_light(light)

camera.type = camera.ORTHO
camera2.set_intrinsic(256, 256, 256, 256)

print(camera2.export_intrinsic())
print(camera2.export_extrinsic())


gui = ti.GUI('Model', camera.res)
gui2 = ti.GUI('Model2', camera2.res)

while gui.running and gui2.running:
    gui.get_event(None)
    gui2.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    gui2.running = not gui2.is_pressed(ti.GUI.ESCAPE)
    model.L2W.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
    gui2.set_image(camera2.img)
    gui2.show()
