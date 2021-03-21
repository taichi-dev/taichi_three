import taichi as ti
import numpy as np
from matplotlib import cm
import os

cmap = cm.get_cmap('magma')


res = 256, 64, 64
#res = 1024, 256, 1
rho = ti.field(float, res)
vel = ti.Vector.field(3, float, res)
img = ti.field(float, (res[0], res[1]))


def load(frame):
    path = f'/tmp/{frame:06d}.npz'
    if not os.path.exists(path):
        return False
    with np.load(path) as data:
        rho.from_numpy(data['rho'])
        vel.from_numpy(data['vel'])
    return True


@ti.func
def color(x, y, z):
    return vel[x, y, z].norm() * 5


@ti.kernel
def render():
    for x, y in img:
        ret = 0.0
        cnt = 0
        for z in range(res[2] // 4, max(1, res[2] * 3 // 4)):
            ret += color(x, y, z)
            cnt += 1
        img[x, y] = ret / cnt


frame = 0
while load(frame):
    print('render for', frame)
    render()
    im = cmap(img.to_numpy())
    ti.imshow(im)
    ti.imwrite(im, f'/tmp/{frame:06d}.png')
    frame += 1
