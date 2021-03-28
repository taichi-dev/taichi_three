import taichi as ti
import numpy as np

ti.init(ti.gpu)


nx, ny = 512, 512


x = ti.field(float, (2, nx, ny // 2))
r = ti.field(float, (2, nx, ny // 2))


@ti.pyfunc
def at(i, j):
    return (i + j) % 2, i, j // 2


@ti.kernel
def solve():
    for t in ti.static(range(2)):
        for i, j in ti.ndrange(nx, ny):
            if (i + j) % 2 != t:
                continue

            xl = x[at(i - 1, j)]
            xr = x[at(i + 1, j)]
            xb = x[at(i, j - 1)]
            xt = x[at(i, j + 1)]
            div = r[at(i, j)]
            xc = (xl + xr + xb + xt - div) * 0.25
            x[at(i, j)] = xc


@ti.kernel
def dump(out: ti.ext_arr()):
    for i, j in ti.ndrange(nx, ny):
        out[i, j] = x[at(i, j)]


r[at(256, 256)] = -2
gui = ti.GUI('jacobi', (nx, ny))
while gui.running and not gui.get_event(gui.ESCAPE):
    for i in range(200):
        solve()
    out = np.empty((nx, ny), dtype=np.float32)
    dump(out)
    gui.set_image(out)
    gui.show()
