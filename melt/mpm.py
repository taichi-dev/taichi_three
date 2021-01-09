from tina.advans import *


@ti.func
def list_subscript(a, i: ti.template()):
    if ti.static(isinstance(i, ti.Expr)):
        k = i
        ret = sum(a) * 0
        for j in ti.static(range(len(a))):
            if k == j:
                ret = a[j]
        return ret
    else:
        return a[i]


@ti.data_oriented
class MPMSolver:
    WATER = 0
    JELLY = 1
    SNOW = 2
    SAND = 3

    def __init__(self, dim=3, n_grid=32, E=100, nu=0.2, gravity=(0, 9.8, 0)):
        self.dim = dim
        self.n_grid = n_grid
        self.dt = 1.8e-2 / self.n_grid
        self.steps = int(1 / (120 * self.dt))

        self.n_particles = self.n_grid ** self.dim // 2**(self.dim - 1)
        self.dx = 1 / self.n_grid

        self.p_rho = 1
        self.p_vol = (self.dx * 0.5)**2
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = tovector(totuple(gravity)[:self.dim])
        self.bound = 3
        self.nu = nu
        self.E = E

        self.mu_0 = E / (2 * (1 + nu))
        self.lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

        self.x = ti.Vector.field(self.dim, float, self.n_particles)
        self.v = ti.Vector.field(self.dim, float, self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, float, self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, float, self.n_particles)
        self.material = ti.field(int, self.n_particles)
        self.Jp = ti.field(float, self.n_particles)

        self.grid_v = ti.Vector.field(self.dim, float, (self.n_grid,) * self.dim)
        self.grid_m = ti.field(float, (self.n_grid,) * self.dim)

        ti.materialize_callback(self.reset)

    @ti.kernel
    def reset(self):
        for i in self.x:
            pos = V(*[ti.random() for i in range(self.dim)]) * 0.3 + 0.5
            self.x[i] = pos
            self.v[i] = ti.Vector.zero(float, self.dim)
            self.F[i] = ti.Matrix.identity(float, self.dim)
            self.material[i] = self.WATER
            self.Jp[i] = 1

    def stencil_range(self):
        return ti.ndrange(*(3,) * self.dim)

    @ti.kernel
    def substep(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(float, self.dim)
            self.grid_m[I] = 0
        ti.block_dim(self.n_grid)
        ti.block_local(self.grid_v)
        ti.block_local(self.grid_m)
        for p in self.x:
            Xp = self.x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            self.F[p] = (ti.Matrix.identity(float, self.dim) + self.dt * self.C[p]) @ self.F[p]
            h = ti.exp(10 * (1.0 - self.Jp[p]))
            if self.material[p] == self.JELLY:
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == self.WATER:
                mu = 0.0
            U, sig, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(self.dim)):
                new_sig = sig[d, d]
                if self.material[p] == self.SNOW:
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
                self.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if self.material[p] == self.WATER:
                self.F[p] = ti.Matrix.identity(float, self.dim) * J**(1 / self.dim)
            elif self.material[p] == self.SNOW:
                self.F[p] = U @ sig @ V.transpose()
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose()
            stress += ti.Matrix.identity(float, self.dim) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4 / self.dx**2) * stress
            affine = stress + self.p_mass * self.C[p]
            for offset in ti.grouped(ti.ndrange(*(3,) * self.dim)):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= list_subscript(w, offset[i])[i]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] /= self.grid_m[I]
            self.grid_v[I] -= self.dt * self.gravity
            cond = I < self.bound and self.grid_v[I] < 0 or I > self.n_grid - self.bound and self.grid_v[I] > 0
            self.grid_v[I] = 0 if cond else self.grid_v[I]
        ti.block_dim(self.n_grid)
        ti.block_local(self.grid_v)
        ti.block_local(self.grid_m)
        for p in self.x:
            Xp = self.x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(float, self.dim)
            new_C = ti.Matrix.zero(float, self.dim, self.dim)
            for offset in ti.grouped(ti.ndrange(*(3,) * self.dim)):
                dpos = offset - fx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= list_subscript(w, offset[i])[i]
                g_v = self.grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.dx
            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * self.v[p]

    def step(self):
        for s in range(self.steps):
            self.substep()

def main():
    ti.init(ti.gpu)
    mpm = MPMSolver()

    scene = tina.Scene()
    pars = tina.SimpleParticles(radius=0.01)
    scene.add_object(pars, tina.Classic())

    #scene = tina.Scene(smoothing=True)
    #mciso = tina.MCISO(int(mpm.n_grid))
    #voxel = tina.Voxelizer(mciso.N, radius=1, weight=18)
    #scene.add_object(tina.MeshTransform(mciso, tina.translate(-1) @ tina.scale(2)))

    wire = tina.MeshToWire(tina.PrimitiveMesh.asset('cube'))
    scene.add_object(wire)

    gui = ti.GUI()
    while gui.running:
        scene.input(gui)
        if gui.is_pressed('r'):
            mpm.reset()
        mpm.step()
        #mciso.clear()
        #voxel.voxelize(mciso.m, mpm.x)
        #mciso.march()
        pars.set_particles(mpm.x.to_numpy() * 2 - 1)
        #colors = np.array(list(map(ti.hex_to_rgb, [0x068587, 0xED553B, 0xEEEEF0])))
        #pars.set_particle_colors(colors[mpm.material.to_numpy()])
        scene.render()
        gui.set_image(scene.img)
        gui.show()

def main2():
    ti.init(ti.gpu)
    mpm = MPMSolver(dim=2, n_grid=128)

    gui = ti.GUI()
    while gui.running:
        mpm.step()
        gui.circles(mpm.x.to_numpy())
        gui.show()

if __name__ == '__main__':
    main()
