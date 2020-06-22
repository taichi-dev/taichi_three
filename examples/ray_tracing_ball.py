import taichi as ti
import taichi_three as t3
from math import cos, sin
from time import time
ti.init(ti.opengl)

scene = t3.SceneRT()
pos = ti.Vector(3, ti.f32, 1)
radius = ti.var(ti.f32, 1)

scene.add_ball(pos, radius)

radius[0] = 0.5
gui = ti.GUI('Ray Tracing Ball')
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    scene.set_light_dir([sin(time() * 2) * sin(time() * 0.6),
        cos(time() * 0.6), cos(time() * 2) * sin(time() * 0.6)])
    scene.render()
    gui.set_image(scene.img)
    gui.show()
