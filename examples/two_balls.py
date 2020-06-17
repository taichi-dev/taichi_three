import taichi as ti
import taichi_three as t3
import time, math
ti.init(ti.opengl)

n = 2
r = t3.Scene((640, 480))
pos = ti.Vector(3, ti.f32, n)
radius = ti.var(ti.f32, n)

r.add_ball(pos, radius)
r.set_light_dir([1, 2, -2])
#r.opt.is_normal_map = True

gui = ti.GUI('Ball', r.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    radius[0] = math.sin(time.time()) * 0.25 + 0.5
    radius[1] = math.cos(time.time() * 1.3) * 0.1 + 0.2
    pos[1] = [0.5, 0.5, 0.0]
    r.render()
    gui.set_image(r.img)
    gui.show()
