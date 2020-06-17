import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
from math import cos, sin
from time import time
ti.init(ti.opengl)

N = 12
dt = 0.01

scene = t3.Scene((640, 480))
pos = ti.Vector(3, ti.f32, N)
vel = ti.Vector(3, ti.f32, N)
radius = ti.var(ti.f32, N)

scene.add_ball(pos, radius)
scene.set_light_dir([1, 1, -1])
#scene.opt.is_normal_map = True

@ti.kernel
def init():
    for i in pos:
        pos[i] = ts.randNDRange(ts.vec3(-1), ts.vec3(1))
        vel[i] = ts.randNDRange(ts.vec3(-1.1), ts.vec3(1.1))
        radius[i] = ts.randRange(0.1, 0.2)

@ti.func
def exchange(v1, m1, v2, m2, disp):
    vel1 = v1.dot(disp)
    vel2 = v2.dot(disp)

    sm1 = ti.sqrt(m1)
    sm2 = ti.sqrt(m2)
    itsm = 1 / ti.sqrt(m1 + m2)

    kero1 = vel1 * sm1
    kero2 = vel2 * sm2

    smd1 =  sm2 * itsm
    smd2 = -sm1 * itsm

    kos = 2 * (kero1 * smd1 + kero2 * smd2)
    kero1 -= kos * smd1
    kero2 -= kos * smd2

    vel1 = kero1 / sm1
    vel2 = kero2 / sm2

    disp *= 0.8

    v1 -= v1.dot(disp) * disp
    v2 -= v2.dot(disp) * disp

    v1 += vel1 * disp
    v2 += vel2 * disp

    return v1, v2

@ti.func
def interact(i, j):
    disp = pos[i] - pos[j]
    disv = vel[i] - vel[j]
    if ts.length(disp) < radius[i] + radius[j] and disp.dot(disv) < 0:
        mass_i = radius[i] ** 3
        mass_j = radius[j] ** 3
        disp = ts.normalize(disp)
        v_i, v_j = exchange(vel[i], mass_i, vel[j], mass_j, disp)
        vel[i], vel[j] = v_i, v_j

@ti.kernel
def substep():
    for i in pos:
        acc = ts.vec(0, -0, 1)
        vel[i] += acc * dt

    for i in pos:
        for j in range(N):
            if i != j:
                interact(i, j)

    for i in pos:
        for j in ti.static(range(3)):
            if vel[i][j] < 0 and pos[i][j] < -1 + radius[j]:
                vel[i][j] *= -0.8
            if vel[i][j] > 0 and pos[i][j] > 1 - radius[j]:
                vel[i][j] *= -0.8

    for i in pos:
        pos[i] += vel[i] * dt

init()
gui = ti.GUI('Ball', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    for i in range(4):
        substep()
    scene.render()
    gui.set_image(scene.img)
    gui.show()
