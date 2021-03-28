import taichi as ti
import numpy as np

ti.init(ti.gpu)


nx, ny = 512, 512


x = ti.field(float)
new_x = ti.field(float)
r = ti.field(float)

ti.root.dense(ti.ij, (nx, ny)).place(x)
#ti.root.dense(ti.ij, (nx // 4, ny // 4)).dense(ti.ij, (4, 4)).place(x)

ti.root.dense(ti.ij, (nx, ny)).place(new_x)
#ti.root.dense(ti.ij, (nx // 4, ny // 4)).dense(ti.ij, (4, 4)).place(new_x)

ti.root.dense(ti.ij, (nx, ny)).place(r)


@ti.func
def sample(f: ti.template(), u, v):
    I = int(ti.Vector([u, v]))
    I = max(0, min(ti.Vector([nx, ny]) - 1, I))
    return f(I.x, I.y)


@ti.func
def calc0(i, j):
    return x[i, j]


@ti.func
def calc1(i, j):
    xc = sample(calc0, i, j)
    xl = sample(calc0, i - 1, j)
    xr = sample(calc0, i + 1, j)
    xb = sample(calc0, i, j - 1)
    xt = sample(calc0, i, j + 1)
    div = r[i, j]
    return (xl + xr + xb + xt - div) * 0.25


@ti.func
def calc2(i, j):
    xc = sample(calc1, i, j)
    xl = sample(calc1, i - 1, j)
    xr = sample(calc1, i + 1, j)
    xb = sample(calc1, i, j - 1)
    xt = sample(calc1, i, j + 1)
    div = r[i, j]
    return (xl + xr + xb + xt - div) * 0.25


@ti.kernel
def solve(x: ti.template(), new_x: ti.template()):
    for i, j in ti.ndrange(nx, ny):
        new_x[i, j] = calc1(i, j)


r[256, 256] = -2
gui = ti.GUI('jacobi', (nx, ny))
while gui.running and not gui.get_event(gui.ESCAPE):
    for i in range(200):
        solve(x, new_x)
        x, new_x = new_x, x
    gui.set_image(x)
    gui.show()
