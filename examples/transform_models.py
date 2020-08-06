import taichi as ti
import taichi_three as t3
import numpy as np
import math, time

ti.init(ti.cpu)

scene = t3.Scene()
monkey = t3.Model(t3.readobj('assets/monkey.obj', scale=0.4))
torus = t3.Model(t3.readobj('assets/torus.obj', scale=0.6))
scene.add_model(monkey)
scene.add_model(torus)

scene.set_light_dir([0.4, -1.5, -1.8])
gui = ti.GUI('Transform', scene.res)
while gui.running:
    gui.get_event()
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    scene.camera.from_mouse(gui)
    monkey.L2W.matrix[None] = t3.rotationZ(angle=time.time())
    torus .L2W.offset[None] = [0, math.cos(time.time()) * 0.5, 0]
    scene.render()
    gui.set_image(scene.img)
    gui.show()
