import taichi as ti
import taichi_three as t3
import numpy as np
import math, time

ti.init(ti.cpu)

scene = t3.Scene()
monkey = t3.Model(obj=t3.readobj('assets/monkey.obj', scale=0.4))
torus = t3.Model(obj=t3.readobj('assets/torus.obj', scale=0.6))
scene.add_model(monkey)
scene.add_model(torus)

camera = t3.Camera()
camera.type = camera.ORTHO
scene.add_camera(camera)

light = t3.Light([0.4, -1.5, 1.8])
scene.add_light(light)
gui = ti.GUI('Transform', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    monkey.L2W.matrix[None] = t3.rotationZ(angle=time.time())
    torus .L2W.offset[None] = [0, math.cos(time.time()) * 0.5, 0]
    scene.render()
    gui.set_image(camera.img)
    gui.show()
