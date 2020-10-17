import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model(t3.Mesh.from_obj('assets/monkey.obj'))
scene.add_model(model)
camera = t3.Camera()
scene.add_camera_d(camera)
buffer = t3.GaussianBlur(t3.FrameBuffer(camera), 8)
scene.add_buffer(buffer)
light = t3.Light([0.4, -1.5, -0.8])
scene.add_light(light)

gui = ti.GUI('Gaussian', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(buffer.img)
    gui.show()
