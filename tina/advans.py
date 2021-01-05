from .common import *


inf = 1e6
eps = 1e-6


def texture_as_field(filename):
    if isinstance(filename, str):
        img_np = ti.imread(filename)
    else:
        img_np = np.array(filename)
    if img_np.dtype == np.uint8:
        img_np = np.float32(img_np / 255)

    img = ti.Vector.field(3, float, img_np.shape[:2])

    @ti.materialize_callback
    def init_texture():
        img.from_numpy(img_np)

    return img


@ti.pyfunc
def aces_tonemap(color):
    # https://zhuanlan.zhihu.com/p/21983679
    return color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14)


@ti.pyfunc
def _film(x):
    # https://zhuanlan.zhihu.com/p/21983679
    A = 0.22
    B = 0.30
    C = 0.10
    D = 0.20
    E = 0.01
    F = 0.30

    return (x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F) - E / F


@ti.pyfunc
def film_tonemap(color):
    return _film(1.6 * color) / _film(11.2)


@ti.pyfunc
def ce_tonemap(color):
    if ti.static(ti.inside_kernel()):
        return 1 - ti.exp(-color)
    else:
        return 1 - np.exp(-color)


@ti.pyfunc
def ce_untonemap(color):
    if ti.static(ti.inside_kernel()):
        # 255 / 256 ~= 0.996
        return -ti.log(1 - clamp(color * 0.996, 0, 1))
    else:
        return -np.log(1 - np.clip(color * 0.996, 0, 1))


@eval('lambda x: x()')
def get_image():
    @ti.kernel
    def _get_image(out: ti.ext_arr(), img: ti.template(), tonemap: ti.template()):
        for I in ti.grouped(img):
            val = tonemap(img[I])
            if ti.static(isinstance(val, ti.Matrix)):
                for k in ti.static(range(val.n)):
                    out[I, k] = val[k]
            else:
                out[I] = val


    def get_image(img, tonemap=lambda x: x):
        shape = img.shape
        if isinstance(img, ti.Matrix):
            shape = shape + (img.n,)
        out = np.empty(shape)
        _get_image(out, img, tonemap)
        return out

    return get_image


@ti.func
def tangentspace(nrm):
    #up = V(0., 1., 0.)
    up = V(233., 666., 512.).normalized()
    '''
    if abs(nrm.y) > 0.9:
        up = V(0., 0., 1.)
        if abs(nrm.z) > 0.9:
            up = V(1., 0., 0.)
    '''
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
    return dir.z, p % 1


@ti.func
def sample_cube(tex: ti.template(), dir):
    I = V(0., 0.)
    dps = 1 - 12 / tex.shape[0]
    eps = 1e-7
    #dir.y, dir.z = dir.z, -dir.y
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
    I = (tex.shape[0] - 1) * I
    return bilerp(tex, I)


@ti.func
def sample_spherical(tex: ti.template(), dir):
    I = V(0., 0.)
    dir.z, dir.y = dir.y, -dir.z
    u, v = unspherical(dir)
    I = (V(*tex.shape) - 1) * V(v, u * 0.5 + 0.5)
    return bilerp(tex, I)


@ti.func
def _inoise(x):
    value = ti.cast(x, ti.u32)
    value = (value ^ 61) ^ (value >> 16)
    value *= 9
    value ^= value << 4
    value *= 0x27d4eb2d
    value ^= value >> 15
    return value


@ti.func
def inoise(x):
    if ti.static(not isinstance(x, ti.Matrix)):
        return _inoise(x)
    index = ti.cast(x, ti.u32)
    value = _inoise(index.entries[0])
    for i in ti.static(index.entries[1:]):
        value = _inoise(i ^ value)
    return value


@ti.func
def noise(x):
    u = inoise(x) >> 1
    return u * (2 / 4294967296)
