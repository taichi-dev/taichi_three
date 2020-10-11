import taichi as ti
import taichi_three as t3
import numpy as np
import time

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model.from_obj(t3.readobj('assets/torus.obj', scale=0.8))
scene.add_model(model)
ball = t3.Model.from_obj(t3.readobj('assets/sphere.obj', scale=0.2))
ball.add_uniform('emission', [1.0, 1.0, 1.0])
scene.add_model(ball)
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[0, 1, 1])
scene.add_camera(camera)
light = t3.PointLight(pos=[0, 1, 0])
scene.add_light(light)
ambient = t3.AmbientLight(0.2)
scene.add_light(ambient)

gui = ti.GUI('Model', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    light.pos[None].y = ti.cos(time.time())
    ball.L2W[None] = t3.translate(light.pos[None].value)
    scene.render()
    gui.set_image(camera.img)
    gui.show()