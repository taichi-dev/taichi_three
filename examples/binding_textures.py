import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model(obj=t3.readobj('assets/torus.obj'),
                 tex=ti.imread('assets/cloth.jpg'))
camera = t3.Camera()
scene.add_model(model)
scene.add_camera(camera)

scene.set_light_dir([0.4, -1.5, 1.8])
gui = ti.GUI('Model', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
