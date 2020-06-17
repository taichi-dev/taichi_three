import taichi as ti
import taichi_three as t3
from math import cos, sin
from time import time
ti.init(ti.opengl)

scene = t3.Scene()
pos = ti.Vector(3, ti.f32, 1)
radius = ti.var(ti.f32, 1)

scene.add_ball(pos, radius)
scene.set_light_dir([1, 2, -2])

radius[0] = 0.5

gui = ti.GUI('Ball')
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    scene.set_light_dir([sin(time()), 1, cos(time())])
    scene.render()
    gui.set_image(scene.img)
    gui.show()
