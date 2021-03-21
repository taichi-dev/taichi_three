import taichi as ti
import numpy as np
import time


ti.init(ti.cpu)


'''D2Q9
directions_np = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0],[0,0,0]])
weights_np = np.array([1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,4.0/9.0])
'''


#'''D3Q15
directions_np = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,1],[-1,-1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1]])
weights_np = np.array([2.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0])
#'''

'''D3Q27
directions_np = np.array([[0,0,0], [1,0,0],[-1,0,0],
		[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,0],[-1,-1,0],[1,0,1],
		[-1,0,-1],[0,1,1],[0,-1,-1],[1,-1,0],[-1,1,0],[1,0,-1],[-1,0,1],
		[0,1,-1],[0,-1,1],[1,1,1],[-1,-1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],
		[-1,1,-1],[-1,1,1],[1,-1,-1]])
weights_np = np.array([8.0/27.0,2.0/27.0,2.0/27.0,2.0/27.0,
			2.0/27.0,2.0/27.0,2.0/27.0, 1.0/54.0,1.0/54.0,1.0/54.0
		,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,1.0/54.0,
	    1.0/54.0,1.0/54.0,1.0/54.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0
        ,1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0])
'''

#res = 512, 128, 1
res = 256, 64, 64
direction_size = len(weights_np)

#niu = 0.01#05
niu = 0.005
tau = 3.0 * niu + 0.5
inv_tau = 1 / tau


rho = ti.field(float, res)
vel = ti.Vector.field(3, float, res)
f_old = ti.field(float, res + (direction_size,))
f_new = ti.field(float, res + (direction_size,))
directions = ti.Vector.field(3, int, direction_size)
weights = ti.field(float, direction_size)


@ti.materialize_callback
def init_velocity_set():
    directions.from_numpy(directions_np)
    weights.from_numpy(weights_np)


@ti.kernel
def initialize():
    for x, y, z in rho:
        rho[x, y, z] = 1
        vel[x, y, z] = ti.Vector.zero(float, 3)

    for x, y, z, i in f_old:
        feq = f_eq(x, y, z, i)
        f_new[x, y, z, i] = feq
        f_old[x, y, z, i] = feq


@ti.func
def f_eq(x, y, z, i):
    eu = vel[x, y, z].dot(directions[i])
    #term = 1 + eu / c_s**2 + eu**2 / (2 * c_s**2)
    #term -= vel[x, y, z].norm_sqr() / (2 * c_s**2)
    uv = vel[x, y, z].norm_sqr()
    term = 1 + 3 * eu + 4.5 * eu**2 - 1.5 * uv
    feq = weights[i] * rho[x, y, z] * term
    return feq


@ti.kernel
def compute_density_momentum_moment():
    for x, y, z in rho:
        '''
        if any(ti.Vector([x, y, z]) == 0):
            continue
        if any(ti.Vector([x, y, z]) == ti.Vector(res) - 1):
            continue
        '''

        new_density = 0.0
        u = ti.Vector.zero(float, 3)
        for i in range(direction_size):
            f = f_new[x, y, z, i]
            f_old[x, y, z, i] = f
            u += f * directions[i]
            new_density += f
        rho[x, y, z] = new_density
        vel[x, y, z] = u / max(new_density, 1e-6)


@ti.kernel
def collide_and_stream():
    for x, y, z, i in f_new:
        xmd, ymd, zmd = (ti.Vector([x, y, z]) - directions[i]) % ti.Vector(res)
        feq = f_eq(xmd, ymd, zmd, i)
        f = (1 - inv_tau) * f_old[xmd, ymd, zmd, i] + inv_tau * feq

        f_new[x, y, z, i] = f


@ti.func
def apply_bc_core(outer, bc_type, bc_value, ibc, jbc, kbc, inb, jnb, knb):
    if (outer == 1):  # handle outer boundary
        if bc_type == 0:
            vel[ibc, jbc, kbc] = bc_value
        elif bc_type == 1:
            vel[ibc, jbc, kbc] = vel[inb, jnb, knb]
    rho[ibc, jbc, kbc] = rho[inb, jnb, knb]
    for l in range(direction_size):
        f_old[ibc,jbc,kbc, l] = f_eq(ibc,jbc,kbc,l) - f_eq(inb,jnb,knb,l) + f_old[inb,jnb,knb, l]


@ti.kernel
def apply_bc():
    for y, z in ti.ndrange((1, res[1] - 1), (1, res[2] - 1)):
    #for y, z in ti.ndrange(res[1], res[2]):
        apply_bc_core(1, 0, [0.1, 0.0, 0.0],
                0, y, z, 1, y, z)
        apply_bc_core(1, 1, [0.0, 0.0, 0.0],
                res[0] - 1, y, z, res[0] - 2, y, z)

    #'''
    for x, z in ti.ndrange(res[0], res[2]):
        apply_bc_core(1, 0, ti.Vector([0.0, 0.0, 0.0]),
                x, res[1] - 1, z, x, res[1] - 2, z)

        apply_bc_core(1, 0, ti.Vector([0.0, 0.0, 0.0]),
                x, 0, z, x, 1, z)

    for x, y in ti.ndrange(res[0], res[1]):
        apply_bc_core(1, 0, ti.Vector([0.0, 0.0, 0.0]),
                x, y, res[2] - 1, x, y, res[2] - 2)

        apply_bc_core(1, 0, ti.Vector([0.0, 0.0, 0.0]),
                x, y, 0, x, y, 1)
    #'''

    for x, y, z in ti.ndrange(*res):
        pos = ti.Vector([x, y, z])
        cpos = ti.Vector(res) / ti.Vector([5, 2, 2])
        cradius = res[1] / 4
        if (pos - cpos).norm_sqr() < cradius**2:
            vel[x, y, z] = ti.Vector.zero(float, 3)
            xnb, ynb, znb = pos + 1 if pos > cpos else pos - 1
            apply_bc_core(0, 0, [0.0, 0.0, 0.0], x, y, z, xnb, ynb, znb)


@ti.kernel
def apply_bc_2d():
    for y, z in ti.ndrange((1, res[1] - 1), 1):
    #for y, z in ti.ndrange(res[1], res[2]):
        apply_bc_core(1, 0, [0.1, 0.0, 0.0],
                0, y, z, 1, y, z)
        apply_bc_core(1, 1, [0.0, 0.0, 0.0],
                res[0] - 1, y, z, res[0] - 2, y, z)

    #'''
    for x, z in ti.ndrange(res[0], 1):
        apply_bc_core(1, 0, ti.Vector([0.0, 0.0, 0.0]),
                x, res[1] - 1, z, x, res[1] - 2, z)

        apply_bc_core(1, 0, ti.Vector([0.0, 0.0, 0.0]),
                x, 0, z, x, 1, z)

    for x, y, z in ti.ndrange(*res):
        pos = ti.Vector([x, y])
        cpos = ti.Vector((res[0], res[1])) / ti.Vector([5, 2])
        cradius = res[1] / 7
        if (pos - cpos).norm_sqr() < cradius**2:
            vel[x, y, z] = ti.Vector.zero(float, 3)
            xnb, ynb = pos + 1 if pos > cpos else pos - 1
            apply_bc_core(0, 0, [0.0, 0.0, 0.0], x, y, z, xnb, ynb, z)


def substep():
    collide_and_stream()
    compute_density_momentum_moment()
    apply_bc()


initialize()
for frame in range(24 * 24):

    print('compute for', frame); t0 = time.time()
    for subs in range(28):
        #print('substep', subs)
        substep()
    print('compute time', time.time() - t0)

    #grid = np.empty(res + (4,), dtype=np.float32)
    #grid[..., 3] = rho.to_numpy()
    #grid[..., :3] = vel.to_numpy()

    print('store for', frame); t0 = time.time()
    np.savez(f'/tmp/{frame:06d}', rho=rho.to_numpy(), vel=vel.to_numpy())
    print('store time', time.time() - t0)
    print('==========')
