@ti.func
def list_subscript(a, i):
    '''magic method to subscript a list with dynamic index'''
    ret = sum(a) * 0
    for j in ti.static(range(len(a))):
        if i == j:
            ret = a[j]
    return ret


@ti.data_oriented
class MPMSolver:
    def __init__(self, dim=3, n_grid=32, E=400, gravity=(0, 9.8, 0)):
        self.dim = dim
        self.n_grid = n_grid
        self.dt = 1.8e-2 / self.n_grid
        self.steps = int(0.02 / self.dt)

        self.n_particles = self.n_grid ** self.dim // 2 ** (self.dim - 1)
        self.dx = 1 / self.n_grid

        self.p_rho = 1
        self.p_vol = (self.dx * 0.5)**2
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = tovector(gravity)
        self.bound = 3
        self.E = E

        self.x = ti.Vector.field(self.dim, float, self.n_particles)
        self.v = ti.Vector.field(self.dim, float, self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, float, self.n_particles)
        self.J = ti.field(float, self.n_particles)

        self.grid_v = ti.Vector.field(self.dim, float, (self.n_grid,) * self.dim)
        self.grid_m = ti.field(float, (self.n_grid,) * self.dim)

        @ti.materialize_callback
        @ti.kernel
        def init_J():
            for i in self.J:
                self.J[i] = 1


    @ti.kernel
    def substep(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = self.grid_v[I] * 0
            self.grid_m[I] = 0
        ti.block_dim(self.n_grid)
        for p in self.x:
            Xp = self.x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            stress = -self.dt * 4 * self.E * self.p_vol * (self.J[p] - 1) / self.dx**2
            affine = ti.Matrix.identity(float, self.dim) * stress + self.p_mass * self.C[p]
            for offset in ti.grouped(ti.ndrange(*[3] * self.dim)):
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
        for p in self.x:
            Xp = self.x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_v = self.v[p] * 0
            new_C = self.C[p] * 0
            for offset in ti.grouped(ti.ndrange(*[3] * self.dim)):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= list_subscript(w, offset[i])[i]
                g_v = self.grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.dx**2
            self.v[p] = new_v
            self.x[p] += self.dt * self.v[p]
            self.J[p] *= 1 + self.dt * new_C.trace()
            self.C[p] = new_C


    def step(self):
        for s in range(self.steps):
            self.substep()
