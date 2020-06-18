import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
from math import cos, sin, tan, pi, tau
from time import time
ti.init(ti.opengl)

N = 12
dt = 0.01

scene = t3.Scene((640, 640))
pos = ti.Vector(3, ti.f32, N)
vel = ti.Vector(3, ti.f32, N)
radius = ti.var(ti.f32, N)
bound = ti.Vector(3, ti.f32, ())

scene.add_ball(pos, radius)
scene.set_light_dir([1, 1, -1])

@ti.kernel
def init():
    for i in pos:
        pos[i] = ts.randNDRange(ts.vec3(-1), ts.vec3(1))
        vel[i] = ts.randNDRange(ts.vec3(-1.1), ts.vec3(1.1))
        radius[i] = ts.randRange(0.1, 0.2)

@ti.func
def interact(i, j):
    disp = pos[i] - pos[j]
    disv = vel[i] - vel[j]
    if disp.norm_sqr() < (radius[i] + radius[j]) ** 2 and disp.dot(disv) < 0:
        mass_i = radius[i] ** 3
        mass_j = radius[j] ** 3
        disp = ts.normalize(disp)
        vel[i], vel[j] = ts.momentumExchange(
                vel[i], vel[j], disp, mass_i, mass_j, 0.8)

@ti.kernel
def substep():
    for i in pos:
        acc = ts.vec(0, -1, 0)
        vel[i] += acc * dt

    for i in pos:
        for j in range(N):
            if i != j:
                interact(i, j)

    for i in pos:
        for j in ti.static(range(3)):
            if vel[i][j] < 0 and pos[i][j] < -1 + radius[i]:
                vel[i][j] *= -0.8
            if vel[i][j] > 0 and pos[i][j] > 1 - radius[i]:
                vel[i][j] *= -0.8

    for i in pos:
        pos[i] += vel[i] * dt

init()
gui = ti.GUI('Ball', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    for i in range(4):
        substep()
    scene.camera.from_mouse(gui, dis=4)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
