import taichi as ti
import numpy as np
import tina

from tina.util.mciso import MCISO, Voxelizer

ti.init(arch=ti.gpu)


#dim, n_grid, steps, dt = 2, 128, 20, 2e-4
#dim, n_grid, steps, dt = 2, 256, 32, 1e-4
dim, n_grid, steps, dt = 3, 32, 25, 4e-4
#dim, n_grid, steps, dt = 3, 64, 25, 2e-4
#dim, n_grid, steps, dt = 3, 128, 25, 8e-5

n_particles = n_grid**dim // 2**(dim - 1)
dx = 1 / n_grid

print(f'n_particles={n_particles}')

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 4
E = 400

x = ti.Vector.field(dim, float, n_particles)
v = ti.Vector.field(dim, float, n_particles)
C = ti.Matrix.field(dim, dim, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
grid_m = ti.field(float, (n_grid, ) * dim)

neighbour = (3, ) * dim


@ti.func
def list_subscript(a, i):
    '''magic method to subscript a list with dynamic index'''
    ret = sum(a) * 0
    for j in ti.static(range(len(a))):
        if i == j:
            ret = a[j]
    return ret


@ti.kernel
def substep():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.zero(grid_v[I])
        grid_m[I] = 0
    ti.block_dim(n_grid)
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix.identity(float, dim) * stress + p_mass * C[p]
        for offset in ti.grouped(ti.ndrange(*neighbour)):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= list_subscript(w, offset[i])[i]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] /= grid_m[I]
        grid_v[I][1] -= dt * gravity
        cond = I < bound and grid_v[I] < 0 or I > n_grid - bound and grid_v[
            I] > 0
        grid_v[I] = 0 if cond else grid_v[I]
    ti.block_dim(n_grid)
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.zero(v[p])
        new_C = ti.zero(C[p])
        for offset in ti.grouped(ti.ndrange(*neighbour)):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= list_subscript(w, offset[i])[i]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C


@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = ti.Vector([ti.random() for i in range(dim)]) * 0.4 + 0.15
        J[i] = 1


mciso = MCISO(n_grid * 2)
voxel = Voxelizer(mciso.N, radius=2, weight=64)

scene = tina.Scene(smoothing=True)

gui = ti.GUI('mciso_mpm3d', scene.res)
scene.init_control(gui, center=[0.5, 0.5, 0.5], radius=1.5)

scene.lighting.clear_lights()
scene.lighting.add_light([-0.4, 1.5, 1.8], color=[0.8, 0.8, 0.8])
scene.lighting.set_ambient_light([0.22, 0.22, 0.22])

scene.add_object(mciso)

init()
while gui.running:
    scene.input(gui)

    if gui.is_pressed('r'):
        init()

    for s in range(steps):
        substep()

    mciso.clear()
    voxel.voxelize(mciso.m, x)
    mciso.march()

    # tina.writeobj(f'/tmp/{gui.frame:04d}.obj', mciso.get_mesh())

    # np.save(f'/tmp/{gui.frame:04d}.npy', x.to_numpy())

    scene.render()

    gui.set_image(scene.img)
    gui.show()
