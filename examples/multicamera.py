import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
texture = ti.imread("assets/cloth.jpg")
model = t3.Model(obj=t3.readobj('assets/monkey.obj', scale=0.6), tex=texture)
scene.add_model(model)
camera = t3.Camera(res=(256, 256))
camera.set(pos=[0, 0, 2.5], target=[0, 0, 0], up=[0, 1, 0])
camera.type = camera.ORTHO
scene.add_camera(camera)

camera2 = t3.Camera()
camera2.set(pos=[0, 1, -2], target=[0, 1, 0], up=[0, 1, 0])
camera2.set_intrinsic(256, 256, 256, 256)
scene.add_camera(camera2)

print(camera2.export_intrinsic())
print(camera2.export_extrinsic())

light = t3.Light([0.4, -1.5, -0.8])
scene.add_light(light)
gui = ti.GUI('Model', camera.res)
gui2 = ti.GUI('Model2', camera2.res)

while gui.running and gui2.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    gui2.running = not gui2.get_event(ti.GUI.ESCAPE)
    model.L2W.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
    gui2.set_image(camera2.img)
    gui2.show()
