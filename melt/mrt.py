import taichi as ti
import numpy as np


res = nx, ny = 512, 512
niu = 1e-4
tau = 3 * niu + 0.5

rho = ti.field(float, res)
vel = ti.Vector.field(2, float, res)
mask = ti.field(float, res)
img = ti.field(float, res)
f_old = ti.Vector.field(9, float, res)
f_new = ti.Vector.field(9, float, res)
W = ti.field(float, 9)
E = ti.field(int, (9, 2))
bc_type = ti.field(int, 4)
bc_value = ti.field(float, (4, 2))

@ti.materialize_callback
def _():
    W.from_numpy(np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
        1.0 / 36.0], np.float32))
    E.from_numpy(np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                        [-1, 1], [-1, -1], [1, -1]], np.int32))
    bc_type.from_numpy(np.array([0, 0, 1, 0]))
    bc_value.from_numpy(np.array([[0.05, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))


@ti.func  # compute equilibrium distribution function
def f_eq(i, j, k):
    eu = float(E[k, 0]) * vel[i, j][0] + float(E[k, 1]) * vel[i, j][1]
    uv = vel[i, j].norm()
    return W[k] * rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * uv)


@ti.kernel
def init():
    for i, j in rho:
        vel[i, j] = [0, 0]
        rho[i, j] = 1
        mask[i, j] = 0
        for k in ti.static(range(9)):
            val = f_eq(i, j, k)
            f_new[i, j][k] = val
            f_old[i, j][k] = val
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            mask[i, j] = 1.0

@ti.func
def to_moment(f):
    M = ti.Matrix([
            [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
            [-4.0, -1.0, -1.0, -1.0, -1.0,  2.0,  2.0,  2.0,  2.0],
            [4.0, -2.0, -2.0, -2.0, -2.0,  1.0,  1.0,  1.0,  1.0],
            [0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0],
            [0.0, -2.0,  0.0,  2.0,  0.0,  1.0, -1.0, -1.0,  1.0],
            [0.0,  0.0,  1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0],
            [0.0,  0.0, -2.0,  0.0,  2.0,  1.0,  1.0, -1.0, -1.0],
            [0.0,  1.0, -1.0,  1.0, -1.0,  0.0,  0.0,  0.0,  0.0],
            [0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  1.0, -1.0]
        ])
    return M @ f

@ti.func
def from_moment(m):
    d = ti.Vector([1./9, 1./36, 1./36, 1./6,
                   1./12, 1./6, 1./12, 1./4, 1./4])
    M = ti.Matrix([
        [1, -4,  4,  0,  0,  0,  0,  0,  0],
        [1, -1, -2,  1, -2,  0,  0,  1,  0],
        [1, -1, -2,  0,  0,  1, -2, -1,  0],
        [1, -1, -2, -1,  2,  0,  0,  1,  0],
        [1, -1, -2,  0,  0, -1,  2, -1,  0],
        [1,  2,  1,  1,  1,  1,  1,  0,  1],
        [1,  2,  1, -1, -1,  1,  1,  0, -1],
        [1,  2,  1, -1, -1, -1, -1,  0,  1],
        [1,  2,  1,  1,  1, -1, -1,  0, -1],
    ])
    return M @ (d * m)


@ti.kernel
def do_collide():
    for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
        feq = ti.Vector([f_eq(i, j, k) for k in range(9)])
        f = f_old[i, j]
        meq = to_moment(feq)
        m = to_moment(f)

        s = ti.Vector([1.0, 1.63, 1.14, 1.0, 1.92, 0.0, 1.92,
                       1 / tau, 1 / tau])
        m += (meq - m) * s

        f_old[i, j] = from_moment(m)


@ti.kernel
def do_stream():
    for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
        bi = ti.static([0, 3, 4, 1, 2, 7, 8, 5, 6])
        for k in ti.static(range(9)):
            ip = i - E[k, 0]
            jp = j - E[k, 1]
            kk = ti.static(bi[k])
            if mask[ip, jp] == 0.0:
                f_new[i, j][k] = f_old[ip, jp][k]
            else:
                f_new[i, j][k] = f_old[i, j][kk]


@ti.kernel
def update_macro_var():
    for i, j in ti.ndrange((1, nx - 1), (1, ny - 1)):
        rho[i, j] = 0.0
        vel[i, j][0] = 0.0
        vel[i, j][1] = 0.0
        for k in ti.static(range(9)):
            f_old[i, j][k] = f_new[i, j][k]
            rho[i, j] += f_new[i, j][k]
            vel[i, j][0] += float(E[k, 0]) * f_new[i, j][k]
            vel[i, j][1] += float(E[k, 1]) * f_new[i, j][k]
        vel[i, j][0] /= rho[i, j]
        vel[i, j][1] /= rho[i, j]


@ti.kernel
def apply_bc():
    for j in ti.ndrange(1, ny - 1):
        # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
        apply_bc_core(1, 0, 0, j, 1, j)

        # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
        apply_bc_core(1, 2, nx - 1, j, nx - 2, j)

    # top and bottom
    for i in ti.ndrange(nx):
        # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
        apply_bc_core(1, 1, i, ny - 1, i, ny - 2)

        # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
        apply_bc_core(1, 3, i, 0, i, 1)


@ti.func
def apply_bc_core(outer, dr, ibc, jbc, inb, jnb):
    if (outer == 1):  # handle outer boundary
        if (bc_type[dr] == 0):
            vel[ibc, jbc][0] = bc_value[dr, 0]
            vel[ibc, jbc][1] = bc_value[dr, 1]
        elif (bc_type[dr] == 1):
            vel[ibc, jbc][0] = vel[inb, jnb][0]
            vel[ibc, jbc][1] = vel[inb, jnb][1]
    rho[ibc, jbc] = rho[inb, jnb]
    for k in ti.static(range(9)):
        # f_old[ibc,jbc][k] = f_eq(ibc,jbc,k) - f_eq(inb,jnb,k) + f_old[inb,jnb][k]
        f_old[ibc, jbc][k] = f_eq(ibc, jbc, k)


@ti.kernel
def update_image():
    for i, j in img:
        img[i, j] = vel[i, j].norm() / 0.05


gui = ti.GUI('lbm', res)
init()
while gui.running and not gui.get_event(gui.ESCAPE):
    do_collide()
    do_stream()
    update_macro_var()
    apply_bc()
    update_image()
    gui.set_image(img)
    gui.show()
