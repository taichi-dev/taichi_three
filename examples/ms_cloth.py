import taichi as ti
import taichi_glsl as tl
import taichi_three as t3

import numpy as np
import math
ti.init(ti.gpu)

### Parameters

N = 128
NN = N, N
W = 1
L = W / N
gravity = 0.5
stiffness = 1600
damping = 2
steps = 30
dt = 5e-4

### Physics

x = ti.Vector(3, ti.f32, NN)
v = ti.Vector(3, ti.f32, NN)
b = ti.Vector(3, ti.f32, NN)
F = ti.Vector(3, ti.f32, NN)


@ti.kernel
def init():
    for i in ti.grouped(x):
        x[i] = tl.vec((i + 0.5) * L - 0.5, 0.8).xzy


links = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
links = [tl.vec(*_) for _ in links]


@ti.kernel
def substep():
    for i in ti.grouped(x):
        acc = x[i] * 0
        for d in ti.static(links):
            disp = x[tl.clamp(i + d, 0, tl.vec(*NN) - 1)] - x[i]
            length = L * float(d).norm()
            acc += disp * (disp.norm() - length) / length**2
        v[i] += stiffness * acc * dt
    for i in ti.grouped(x):
        v[i].y -= gravity * dt
        v[i] = tl.ballBoundReflect(x[i], v[i], tl.vec(+0.0, +0.2, -0.0), 0.4, 6)
    for i in ti.grouped(x):
        v[i] *= math.exp(-damping * dt)
        x[i] += dt * v[i]


### Rendering GUI

scene = t3.Scene()
model = t3.Model(f_n=(N - 1)**2 * 2, vi_n=N**2, vt_n=N**2,
                 tex=ti.imread('assets/cloth.jpg'))
scene.add_model(model)


@ti.kernel
def init_display():
    for i_ in ti.grouped(ti.ndrange(N - 1, N - 1)):
        i = i_
        a = i.dot(tl.vec(N, 1))
        i.x += 1
        b = i.dot(tl.vec(N, 1))
        i.y += 1
        c = i.dot(tl.vec(N, 1))
        i.x -= 1
        d = i.dot(tl.vec(N, 1))
        i.y -= 1
        model.faces[a * 2 + 0] = tl.mat([a, c, b], [a, c, b], [0, 0, 0]).transpose()
        model.faces[a * 2 + 1] = tl.mat([a, d, c], [a, d, c], [0, 0, 0]).transpose()
    for i in ti.grouped(x):
        j = i.dot(tl.vec(N, 1))
        model.vt[j] = tl.D.yx + i.xY / N


@ti.kernel
def update_display():
    for i in ti.grouped(x):
        j = i.dot(tl.vec(N, 1))
        model.vi[j] = x[i]


init()
init_display()
scene.set_light_dir([0.4, -1.5, -1.8])

with ti.GUI('Mass Spring') as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        if not gui.is_pressed(gui.SPACE):
            for i in range(steps):
                substep()
            update_display()

        mx, my = gui.get_cursor_pos()
        scene.camera.from_mouse((mx, my * 0.5 + 1e-3))

        scene.render()
        gui.set_image(scene.img)
        gui.show()
