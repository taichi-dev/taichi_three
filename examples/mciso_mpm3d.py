import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(arch=ti.cuda)

#dim, n_grid, steps, dt = 2, 128, 20, 2e-4
#dim, n_grid, steps, dt = 2, 256, 32, 1e-4
dim, n_grid, steps, dt = 3, 32, 25, 4e-4
#dim, n_grid, steps, dt = 3, 64, 25, 2e-4
#dim, n_grid, steps, dt = 3, 128, 25, 8e-5

n_particles = n_grid**dim // 2**(dim - 1)
dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = ti.Vector.field(dim, float, n_particles)
v = ti.Vector.field(dim, float, n_particles)
C = ti.Matrix.field(dim, dim, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(dim, float, (n_grid, ) * dim)
grid_m = ti.field(float, (n_grid, ) * dim)

neighbour = (3, ) * dim


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
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
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
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
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


mciso = t3.MCISO(n_grid * 2)
voxel = t3.Voxelizer(mciso.N)

scene = t3.Scene()
mesh = t3.DynamicMesh(n_faces=mciso.N_res, n_pos=mciso.N_res, n_nrm=mciso.N_res)
model = t3.Model(mesh)
#model.material = t3.MaterialVisualizeNormal()
scene.add_model(model)
camera = t3.Camera()
scene.add_camera(camera)
scene.add_light(t3.Light([0.4, -1.5, -1.8], 0.8))
scene.add_light(t3.AmbientLight(0.22))

@ti.kernel
def update_mesh():
    mesh.n_faces[None] = mciso.Js_n[None]
    for i in range(mciso.Js_n[None]):
        for t in ti.static(range(3)):
            mesh.faces[i][t, 0] = mciso.Jts[i][t]
            mesh.faces[i][t, 2] = mciso.Jts[i][t]
        mesh.pos[i] = (mciso.vs[i] + 0.5) / mciso.N * 2 - 1
        mesh.nrm[i] = mciso.ns[i]

init()
gui = ti.GUI('MPM3D', camera.res)
while gui.running:
    for s in range(steps):
        substep()

    gui.get_event(None)
    camera.from_mouse(gui)
    mciso.clear()
    voxel.voxelize(mciso.m, x)
    mciso.march()

    update_mesh()
    scene.render()
    gui.set_image(camera.img)
    gui.show()
