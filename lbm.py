import taichi as ti
import numpy as np
from matplotlib import cm
import math


ti.init(ti.cuda)


directions_np = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,1],[-1,-1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1]])
weights_np = np.array([2.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0])

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

resolution = (64,) * 3
sonic_speed = 2 / math.sqrt(3)
direction_size = len(weights_np)
cmap = cm.get_cmap('magma')


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
    for x, y, z, i in previous_particle_distributions:
        dis = (ti.Vector([x, y, z]) - ti.Vector(resolution) / 2) / 5
        fac = 64 * ti.exp(-dis.norm_sqr())

        previous_particle_distributions[x, y, z, i] = weights[i] * fac
        particle_distributions[x, y, z, i] = weights[i] * fac


@ti.func
def calculate_feq(x, y, z, i):
    dp = velocity_field[x, y, z].dot(directions[i])
    term = 1 + dp / sonic_speed**2 + dp**2 / (2 * sonic_speed**2)
    term -= velocity_field[x, y, z].norm_sqr() / (2 * sonic_speed**2)
    feq = weights[i] * density_field[x, y, z] * term
    return feq


@ti.kernel
def compute_density_momentum_moment():
    for x, y, z in density_field:
        new_density = 1e-6
        u = ti.Vector.zero(float, 3)
        for i in range(direction_size):
            new_density += particle_distributions[x, y, z, i]
            u += particle_distributions[x, y, z, i] * directions[i]
        density_field[x, y, z] = new_density
        velocity_field[x, y, z] = u / new_density


@ti.kernel
def stream():
    for x, y, z, i in particle_distributions:
        res = ti.Vector(resolution)
        xmd, ymd, zmd = (res + ti.Vector([x, y, z]) - directions[i]) % res
        particle_distributions[x, y, z, i] = previous_particle_distributions[xmd, ymd, zmd, i]


@ti.kernel
def collision():
    for x, y, z, i in particle_distributions:
        feq = calculate_feq(x, y, z, i)
        f = (1 - 1 / math.tau) * particle_distributions[x, y, z, i] + feq / math.tau
        previous_particle_distributions[x, y, z, i] = f


def substep():
    compute_density_momentum_moment()
    collision()
    stream()


@ti.kernel
def render():
    for x, y in rendered_image:
        result = 0.0
        for z in range(resolution[2]):
            result += density_field[x, y, z]
        result /= resolution[2]
        rendered_image[x, y] = result


initialize()
gui = ti.GUI('LBM')
while gui.running and not gui.get_event(gui.ESCAPE):
    if gui.is_pressed('r'):
        initialize()
    substep()
    render()
    img = rendered_image.to_numpy()
    img_min, img_max = img.min(), img.max()
    #img = (img - img_min) / (img_max - img_min + 1e-6)
    print(f'{img_min:.02f} {img_max:.02f}')
    gui.set_image(ti.imresize(cmap(img), 512))
    gui.show()
