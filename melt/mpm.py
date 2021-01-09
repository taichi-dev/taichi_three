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

    def __init__(self, dim=3, E=400, nu=0.2, gravity=(0, 9.8, 0)):
        self.dim = dim
        self.n_grid = 32
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
        sin_phi = ti.sin(np.radians(45))
        self.alpha = ti.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

        self.x = ti.Vector.field(self.dim, float)
        self.v = ti.Vector.field(self.dim, float)
        self.C = ti.Matrix.field(self.dim, self.dim, float)
        self.F = ti.Matrix.field(self.dim, self.dim, float)
        self.material = ti.field(int)
        self.Jp = ti.field(float)

        indices = ti.ij if self.dim == 2 else ti.ijk

        grid_size = 4096
        grid_block_size = 128
        leaf_block_size = 16 if self.dim == 2 else 8
        self.grid = ti.root.pointer(indices, grid_size // grid_block_size)
        block = self.grid.pointer(indices, grid_block_size // leaf_block_size)

        self.offset = (-grid_size // 2,) * self.dim
        def block_component(c):
            block.dense(indices, leaf_block_size).place(c, offset=self.offset)

        self.grid_v = ti.Vector.field(self.dim, float)
        self.grid_m = ti.field(float)

        block_component(self.grid_m)
        for v in self.grid_v.entries:
            block_component(v)

        self.pid = ti.field(int)
        block.dynamic(ti.indices(self.dim), 2**20,
                chunk_size=leaf_block_size**self.dim * 8).place(
                        self.pid, offset=self.offset + (0,))

        max_num_particles = 2**20
        self.particle = ti.root.dynamic(ti.i, max_num_particles)
        for c in [self.x, self.v, self.C, self.F, self.material, self.Jp]:
            self.particle.place(c)
        self.particle_num = ti.field(int, ())

        ti.materialize_callback(self.reset)

    @ti.kernel
    def reset(self):
        for i in range(self.n_particles):
            pos = V(*[ti.random() for i in range(self.dim)]) * 0.3 + 0.5
            vel = ti.Vector.zero(float, self.dim)
            self.seed_particle(i, pos, vel, self.WATER)

    @ti.func
    def seed_particle(self, i, pos, vel, material):
        self.x[i] = pos
        self.v[i] = vel
        self.F[i] = ti.Matrix.identity(float, self.dim)
        self.material[i] = material
        if material == self.SAND:
            self.Jp[i] = 0
        else:
            self.Jp[i] = 1

    def stencil_range(self):
        return ti.ndrange(*(3,) * self.dim)

    @ti.kernel
    def build_pid(self):
        ti.block_dim(64)
        for p in self.particle:
            base = ifloor(self.x[p] / self.dx - 0.5)
            ti.append(self.pid.parent(), base - tovector(self.offset), p)

    @ti.kernel
    def p2g(self):
        ti.no_activate(self.particle)
        ti.block_dim(256)
        ti.block_local(self.grid_v)
        ti.block_local(self.grid_m)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            Xp = self.x[p] / self.dx
            base = ifloor(Xp - 0.5)
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
            if self.material[p] != self.SAND:
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
            stress = ti.Matrix.identity(float, self.dim)
            if self.material[p] != self.SAND:
                stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose()
                stress += ti.Matrix.identity(float, self.dim) * la * J * (J - 1)
            else:
                sig = self.sand_projection(sig, p)
                self.F[p] = U @ sig @ V.transpose()
                log_sig_sum = 0.0
                center = ti.Matrix.zero(float, self.dim, self.dim)
                for i in ti.static(range(self.dim)):
                    log_sig = ti.log(sig[i, i])
                    center[i, i] = 2.0 * self.mu_0 * log_sig / sig[i, i]
                    log_sig_sum += log_sig
                for i in ti.static(range(self.dim)):
                    center[i, i] += self.lambda_0 * log_sig_sum / sig[i, i]
                stress = U @ center @ V.transpose() @ self.F[p].transpose()

            stress = (-self.dt * self.p_vol * 4 / self.dx**2) * stress
            affine = stress + self.p_mass * self.C[p]
            for offset in ti.grouped(ti.ndrange(*(3,) * self.dim)):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= list_subscript(w, offset[i])[i]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.kernel
    def grid_op(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] /= self.grid_m[I]
            self.grid_v[I] -= self.dt * self.gravity
            cond1 = I < self.bound and self.grid_v[I] < 0
            cond2 = I > self.n_grid - self.bound and self.grid_v[I] > 0
            self.grid_v[I] = 0 if cond1 or cond2 else self.grid_v[I]

    @ti.kernel
    def g2p(self):
        ti.no_activate(self.particle)
        ti.block_dim(256)
        ti.block_local(self.grid_v)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            Xp = self.x[p] / self.dx
            base = ifloor(Xp - 0.5)
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

    @ti.func
    def sand_projection(self, sigma, p):
        sigma_out = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        epsilon = ti.Vector.zero(ti.f32, self.dim)
        for i in ti.static(range(self.dim)):
            epsilon[i] = ti.log(max(abs(sigma[i, i]), 1e-4))
            sigma_out[i, i] = 1
        tr = epsilon.sum() + self.Jp[p]
        epsilon_hat = epsilon - tr / self.dim
        epsilon_hat_norm = epsilon_hat.norm() + 1e-20
        if tr >= 0.0:
            self.Jp[p] = tr
        else:
            self.Jp[p] = 0.0
            delta_gamma = epsilon_hat_norm + (
                self.dim * self.lambda_0 +
                2 * self.mu_0) / (2 * self.mu_0) * tr * self.alpha
            for i in ti.static(range(self.dim)):
                sigma_out[i, i] = ti.exp(epsilon[i] - max(0, delta_gamma) /
                                         epsilon_hat_norm * epsilon_hat[i])

        return sigma_out

    @ti.kernel
    def get_num_particles(self) -> int:
        return ti.length(self.particle, [])

    @ti.kernel
    def _get_particle_pos(self, out: ti.ext_arr()):
        for i in range(out.shape[0]):
            for k in ti.static(range(3)):
                out[i, k] = self.x[i][k]

    def get_particle_pos(self):
        num = self.get_num_particles()
        out = np.empty((num, 3), dtype=np.float32)
        self._get_particle_pos(out)
        return out

    def step(self):
        for s in range(self.steps):
            self.grid.deactivate_all()
            self.build_pid()
            self.p2g()
            self.grid_op()
            self.g2p()

def main():
    ti.init(ti.gpu, make_block_local=False, kernel_profiler=True)
    mpm = MPMSolver(dim=3)

    scene = tina.Scene()
    pars = tina.SimpleParticles(radius=0.01)
    scene.add_object(pars, tina.Classic())

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
        pars.set_particles(mpm.get_particle_pos() * 2 - 1)
        #colors = np.array(list(map(ti.hex_to_rgb, [0x068587, 0xED553B, 0xEEEEF0])))
        #pars.set_particle_colors(colors[mpm.material.to_numpy()])
        scene.render()
        gui.set_image(scene.img)
        gui.show()

    ti.kernel_profiler_print()

def main2():
    ti.init(ti.gpu)
    mpm = MPMSolver(dim=2)

    gui = ti.GUI()
    while gui.running:
        mpm.step()
        gui.circles(mpm.get_particle_pos())
        gui.show()

if __name__ == '__main__':
    main()
