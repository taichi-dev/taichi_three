import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
import time, math
ti.init(ti.opengl)

n = 2

r = t3.Scene()
pos = ti.Vector(3, ti.f32, n)
radius = ti.var(ti.f32, n)

r.add_ball(pos, radius)

gui = ti.GUI('Ball')
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)

    radius[0] = math.sin(time.time() % 20000) * 0.25 + 0.5
    radius[1] = math.cos(time.time() % 20000 * 1.3) * 0.1 + 0.2
    pos[1] = [0.5, 0.5, 0.0]
    r.render()
    gui.set_image(r.img)
    gui.show()
