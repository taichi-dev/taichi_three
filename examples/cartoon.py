import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
from math import cos, sin
from time import time
ti.init(ti.opengl)

scene = t3.Scene()
pos = ti.Vector(3, ti.f32, 1)
radius = ti.var(ti.f32, 1)
cartoon_level = ti.var(ti.f32, ())

scene.add_ball(pos, radius)
scene.set_light_dir([1, 2, -2])

@ti.func
def my_render_func(pos, normal, dir, light_dir):
    n = cartoon_level[None]
    refl_dir = ts.reflect(light_dir, normal)
    #refl_dir = ts.mix(light_dir, -dir, 0.5)
    NoL = pow(ts.dot(normal, refl_dir), 12)
    NoL = ts.mix(NoL, ts.dot(normal, light_dir) * 0.5 + 0.5, 0.5)
    strength = 0.2
    if any(normal):
        strength = ti.floor(max(0, NoL * n + 0.5)) / n
    return ts.vec3(strength)

scene.opt.render_func = my_render_func

radius[0] = 0.5
gui = ti.GUI('Ball')
while gui.running:
    cartoon_level[None] = 2 + (sin(time() * 0.4) * 0.5 + 0.5) * 22
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
