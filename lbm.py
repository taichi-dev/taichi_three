import taichi as ti
import numpy as np
from matplotlib import cm
import math


ti.init(ti.cuda)


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

resolution = 64, 32, 32
c_s = 2 / math.sqrt(3)
direction_size = len(weights_np)
cmap = cm.get_cmap('magma')

niu = 0.005
tau = 3.0 * niu + 0.5
inv_tau = 1 / tau


density_field = ti.field(float, resolution)
velocity_field = ti.Vector.field(3, float, resolution)
previous_particle_distributions = ti.field(float, resolution + (direction_size,))
particle_distributions = ti.field(float, resolution + (direction_size,))
directions = ti.Vector.field(3, int, direction_size)
weights = ti.field(float, direction_size)
reverse_indices = ti.field(int, direction_size)
rendered_image = ti.field(float, (resolution[0], resolution[1]))


@ti.materialize_callback
def init_velocity_set():
    directions.from_numpy(directions_np)
    weights.from_numpy(weights_np)
    lookup_reverse()


@ti.kernel
def lookup_reverse():
    for i, j in ti.ndrange(direction_size, direction_size):
        if all(directions[i] == -directions[j]):
            reverse_indices[i] = j


@ti.kernel
def initialize():
    for x, y, z in density_field:
        density_field[x, y, z] = 1
        velocity_field[x, y, z] = ti.Vector.zero(float, 3)

    for x, y, z, i in previous_particle_distributions:
        feq = calculate_feq(x, y, z, i)
        particle_distributions[x, y, z, i] = feq
        previous_particle_distributions[x, y, z, i] = feq


@ti.func
def calculate_feq(x, y, z, i):
    dp = velocity_field[x, y, z].dot(directions[i])
    term = 1 + dp / c_s**2 + dp**2 / (2 * c_s**2)
    term -= velocity_field[x, y, z].norm_sqr() / (2 * c_s**2)
    feq = weights[i] * density_field[x, y, z] * term
    return feq


@ti.kernel
def compute_density_momentum_moment():
    for x, y, z in density_field:
        if any(ti.Vector([x, y, z]) == 0):
            continue
        if any(ti.Vector([x, y, z]) == ti.Vector(resolution) - 1):
            continue

        new_density = 0.0
        u = ti.Vector.zero(float, 3)
        for i in range(direction_size):
            f = particle_distributions[x, y, z, i]
            previous_particle_distributions[x, y, z, i] = f
            u += f * directions[i]
            new_density += f
        density_field[x, y, z] = new_density
        velocity_field[x, y, z] = u / max(new_density, 1e-6)


@ti.kernel
def collide_and_stream():
    for x, y, z, i in particle_distributions:
        '''
        if any(ti.Vector([x, y, z]) == 0):
            continue
        if any(ti.Vector([x, y, z]) == ti.Vector(resolution) - 1):
            continue
        '''

        xmd, ymd, zmd = (ti.Vector([x, y, z]) - directions[i]) % ti.Vector(resolution)
        feq = calculate_feq(xmd, ymd, zmd, i)
        f = (1 - inv_tau) * previous_particle_distributions[xmd, ymd, zmd, i] + inv_tau * feq

        particle_distributions[x, y, z, i] = f


@ti.kernel
def collision():
    for x, y, z, i in particle_distributions:
        feq = calculate_feq(x, y, z, i)
        f = (1 - inv_tau) * previous_particle_distributions[x, y, z, i] + inv_tau * feq
        particle_distributions[x, y, z, i] = f


@ti.kernel
def stream():
    for x, y, z, i in particle_distributions:
        #'''periodic
        xmd, ymd, zmd = (ti.Vector([x, y, z]) - directions[i]) % ti.Vector(resolution)
        particle_distributions[x, y, z, i] = previous_particle_distributions[xmd, ymd, zmd, i]
        #'''

        '''couette
        j = reverse_indices[i]
        if y == 0 and directions[i].y == 1:
            particle_distributions[x, y, z, i] = previous_particle_distributions[x, y, z, j]
        elif y == resolution[1] - 1 and directions[i].y == -1:
            u_max = 1.0
            particle_distributions[x, y, z, i] = previous_particle_distributions[x, y, z, j] + directions[i].x * 2 * weights[i] / c_s**2 * u_max
        else:
            xmd, _, zmd = (ti.Vector([x, y, z]) - directions[i]) % ti.Vector(resolution)
            ymd = y - directions[i].y
            particle_distributions[x, y, z, i] = previous_particle_distributions[xmd, ymd, zmd, i]
        '''


@ti.func
def apply_bc_core(outer, bc_type, bc_value, ibc, jbc, kbc, inb, jnb, knb):
    if (outer == 1):  # handle outer boundary
        if bc_type == 0:
            velocity_field[ibc, jbc, kbc] = bc_value
        elif bc_type == 1:
            velocity_field[ibc, jbc, kbc] = velocity_field[inb, jnb, knb]
    density_field[ibc, jbc, kbc] = density_field[inb, jnb, knb]
    for l in range(direction_size):
        previous_particle_distributions[ibc,jbc,kbc, l] = calculate_feq(ibc,jbc,kbc,l) - calculate_feq(inb,jnb,knb,l) + previous_particle_distributions[inb,jnb,knb, l]


@ti.kernel
def apply_bc():
    #for y, z in ti.ndrange((1, resolution[1] - 1), (1, resolution[2] - 1)):
    for y, z in ti.ndrange(resolution[1], resolution[2]):
        apply_bc_core(1, 0, [0.1, 0.0, 0.0],
                0, y, z, 1, y, z)
        apply_bc_core(1, 1, [0.0, 0.0, 0.0],
                resolution[0] - 1, y, z, resolution[0] - 2, y, z)

    for x, y, z in ti.ndrange(*resolution):
        pos = ti.Vector([x, y, z])
        cpos = ti.Vector(resolution) / 2
        if (pos - cpos).norm() < 10:
            velocity_field[x, y, z] = ti.Vector.zero(float, 3)
            xnb, ynb, znb = pos + 1 if pos > cpos else pos - 1
            apply_bc_core(0, 0, [0.0, 0.0, 0.0], x, y, z, xnb, ynb, znb)

    '''
    for x, z in ti.ndrange(resolution[0], resolution[2]):
        apply_bc_core(1, 0, ti.Vector([0.0, 0.0, 0.0]),
                x, resolution[1] - 1, z, x, resolution[1] - 2, z)

        apply_bc_core(1, 0, ti.Vector([0.0, 0.0, 0.0]),
                x, 0, z, x, 1, z)

    for x, y in ti.ndrange(resolution[0], resolution[1]):
        apply_bc_core(1, 0, ti.Vector([0.0, 0.0, 0.0]),
                x, y, resolution[2] - 1, x, y, resolution[2] - 2)

        apply_bc_core(1, 0, ti.Vector([0.0, 0.0, 0.0]),
                x, y, 0, x, y, 1)
    '''


def substep():
    #collision()
    #stream()
    collide_and_stream()
    compute_density_momentum_moment()
    apply_bc()


@ti.kernel
def render():
    for x, y in rendered_image:
        result = 0.0
        for z in range(resolution[2]):
            result += velocity_field[x, y, z].norm() / 0.05
            #result += density_field[x, y, z] * 0.5
        result /= resolution[2]
        rendered_image[x, y] = result


initialize()
gui = ti.GUI('LBM', (512, 256))
gui.fps_limit = 24
while gui.running and not gui.get_event(gui.ESCAPE):
    if gui.is_pressed('r'):
        initialize()
    for s in range(32):
        substep()
    render()
    img = rendered_image.to_numpy()
    img_min, img_max = img.min(), img.max()
    #img = (img - img_min) / (img_max - img_min + 1e-6)
    print(f'{img_min:.02f} {img_max:.02f}')
    gui.set_image(ti.imresize(cmap(img), *gui.res))
    gui.show()
