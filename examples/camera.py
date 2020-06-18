import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
from math import cos, sin, tan, pi, tau
ti.init(ti.opengl)

scene = t3.Scene()
pos = ti.Vector(3, ti.f32, 3)
radius = ti.var(ti.f32, 3)

scene.add_ball(pos, radius)
scene.set_light_dir([0, 0, -1])

radius[0] = 0.3
radius[1] = 0.2
radius[2] = 0.2
pos[1] = [-0.3, 0.3, 0]
pos[2] = [+0.3, 0.3, 0]

gui = ti.GUI('Camera')
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    a, t = gui.get_cursor_pos()
    if a != 0 and t != 0:
        a, t = a * tau - pi, t * pi - pi / 2
    scene.set_camera([1 * sin(a), 1 * tan(t), -1 * cos(a)])
    scene.render()
    gui.set_image(scene.img)
    gui.show()
