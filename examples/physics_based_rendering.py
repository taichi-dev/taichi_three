import taichi as ti
import taichi_three as t3
import numpy as np
import time

ti.init(ti.cpu)

scene = t3.Scene()
obj = t3.readobj('assets/torus.obj', scale=0.8)
model = t3.Model.from_obj(obj)
model.add_texture('color', ti.imread('assets/cloth.jpg'))
model.add_texture('roughness', ti.imread('assets/pattern.jpg'))
model.add_texture('metallic', np.array([[0.7]]))
scene.add_model(model)
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[0, 1, -1.5])
scene.add_camera(camera)
light = t3.Light([0.4, -1.5, 1.8])
scene.add_light(light)
ambient = t3.AmbientLight(0.54)
scene.add_light(ambient)

gui = ti.GUI('Model', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    model.L2W[None] = t3.rotateX(angle=time.time())
    scene.render()
    gui.set_image(camera.img)
    gui.show()
