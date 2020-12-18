from .common import *


def texture_as_field(filename):
    if isinstance(filename, str):
        img_np = np.float32(ti.imread(filename) / 255)
    else:
        img_np = np.array(filename)
    img = ti.Vector.field(3, float, img_np.shape[:2])

    @ti.materialize_callback
    def init_texture():
        img.from_numpy(img_np)

    return img


@ti.pyfunc
def aces_tonemap(color):
    # https://zhuanlan.zhihu.com/p/21983679
    return color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14)


@ti.func
def ce_tonemap(color):
    return 1 - ti.exp(-color)


@ti.func
def tangentspace(nrm):
    up = V(0., 1., 0.)
    bitan = nrm.cross(up).normalized()
    tan = bitan.cross(nrm)
    return ti.Matrix.cols([tan, bitan, nrm])


@ti.func
def spherical(h, p):
    unit = V(ti.cos(p * ti.tau), ti.sin(p * ti.tau))
    dir = V23(ti.sqrt(1 - h**2) * unit, h)
    return dir


@ti.func
def unspherical(dir):
    p = ti.atan2(dir.y, dir.x) / ti.tau
    return dir.z, p


@ti.func
def sample_cube(tex: ti.template(), dir):
    I = V(0., 0.)
    eps = 1e-5
    dps = 1 - 12 / tex.shape[0]
    dir.y, dir.z = dir.z, -dir.y
    if dir.z >= 0 and dir.z >= abs(dir.y) - eps and dir.z >= abs(dir.x) - eps:
        I = V(3 / 8, 3 / 8) + V(dir.x, dir.y) / dir.z / 8 * dps
    if dir.z <= 0 and -dir.z >= abs(dir.y) - eps and -dir.z >= abs(dir.x) - eps:
        I = V(7 / 8, 3 / 8) + V(-dir.x, dir.y) / -dir.z / 8 * dps
    if dir.x <= 0 and -dir.x >= abs(dir.y) - eps and -dir.x >= abs(dir.z) - eps:
        I = V(1 / 8, 3 / 8) + V(dir.z, dir.y) / -dir.x / 8 * dps
    if dir.x >= 0 and dir.x >= abs(dir.y) - eps and dir.x >= abs(dir.z) - eps:
        I = V(5 / 8, 3 / 8) + V(-dir.z, dir.y) / dir.x / 8 * dps
    if dir.y >= 0 and dir.y >= abs(dir.x) - eps and dir.y >= abs(dir.z) - eps:
        I = V(3 / 8, 5 / 8) + V(dir.x, -dir.z) / dir.y / 8 * dps
    if dir.y <= 0 and -dir.y >= abs(dir.x) - eps and -dir.y >= abs(dir.z) - eps:
        I = V(3 / 8, 1 / 8) + V(dir.x, dir.z) / -dir.y / 8 * dps
    I = V(tex.shape[0], tex.shape[0]) * I
    return bilerp(tex, I)
