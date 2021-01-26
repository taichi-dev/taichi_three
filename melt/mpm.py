from tina.advans import *
from voxelizer import MeshVoxelizer


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

    def __init__(self, res, size=1, dt_scale=1, E_scale=1, nu_scale=1):
        self.res = tovector(res)
        self.dim = len(self.res)
        assert self.dim in [2, 3]
        self.dx = size / self.res.x
        self.dt = 2e-2 * self.dx / size * dt_scale
        self.steps = int(1 / (120 * self.dt))

        self.p_rho = 1e3
        self.p_vol = self.dx**self.dim
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = tovector((0, 9.8, 0)[:self.dim])
        self.E = 1e6 * size * E_scale
        self.nu = 0.2 * nu_scale

        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        sin_phi = ti.sin(np.radians(45))
        self.alpha = ti.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

        self.x = ti.Vector.field(self.dim, float)
        self.v = ti.Vector.field(self.dim, float)
        self.C = ti.Matrix.field(self.dim, self.dim, float)
        self.F = ti.Matrix.field(self.dim, self.dim, float)
        self.material = ti.field(int)
        self.Jp = ti.field(float)

        indices = ti.ij if self.dim == 2 else ti.ijk

        grid_size = 1024
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

        max_num_particles = 2**27
        self.particle = ti.root.dynamic(ti.i, max_num_particles, 2**20)
        for c in [self.x, self.v, self.C, self.F, self.material, self.Jp]:
            self.particle.place(c)
        self.particle_num = ti.field(int, ())

        self.grid_postprocess = []

    @ti.kernel
    def seed_volume(self, vox: ti.template(),
            bmin: ti.template(), bmax: ti.template(),
            vel: ti.template(), material: ti.template(),
            ppg: ti.template()):
        for I in ti.grouped(vox.voxels):
            if vox.voxels[I] <= 0:
                continue
            scale = Vprod((bmax - bmin) * self.res / vox.res)
            ppv = int(ppg * scale + ti.random())
            n = ti.atomic_add(self.particle_num[None], ppv)
            for i in range(ppv):
                bias = V(ti.random(), ti.random(), ti.random())
                pos = lerp((I + bias) / vox.res, bmin, bmax)
                self.seed_particle(n + i, pos, vel, material)

    @ti.func
    def seed_particle(self, i, pos, vel, material):
        self.x[i] = pos
        self.v[i] = vel
        self.F[i] = ti.Matrix.identity(float, self.dim)
        self.C[i] = ti.Matrix.zero(float, self.dim, self.dim)
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
            h = ti.exp(10 * (1 - self.Jp[p]))
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
    def grid_normalize(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] /= self.grid_m[I]
            self.grid_v[I] -= self.dt * self.gravity

    @ti.kernel
    def grid_boundary(self):
        for I in ti.grouped(self.grid_m):
            cond1 = I < -self.res and self.grid_v[I] < 0
            cond2 = I > self.res and self.grid_v[I] > 0
            self.grid_v[I] = 0 if cond1 or cond2 else self.grid_v[I]

    def collide_sphere(self, center, radius):
        center = V(*center)

        @ti.kernel
        def collide():
            for I in ti.grouped(self.grid_m):
                offset = I * self.dx - center
                if offset.norm_sqr() < radius**2:
                    self.grid_v[I] *= 0

        self.grid_postprocess.append(collide)

    def collide_volume(self, vox, bmin, bmax):
        @ti.kernel
        def collide():
            for I in ti.grouped(self.grid_m):
                pos = I / self.res
                if all(bmin <= pos <= bmax):
                    J = vox.res * unlerp(pos, bmin, bmax)
                    #rho = trilerp(vox.voxels, J)
                    rho = vox.voxels[ifloor(J)]
                    if rho >= 0.5:
                        self.grid_v[I] *= 0

        self.grid_postprocess.append(collide)

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
            for k in ti.static(range(self.dim)):
                out[i, k] = self.x[i][k]

    def get_particle_pos(self):
        num = self.get_num_particles()
        out = np.empty((num, self.dim), dtype=np.float32)
        self._get_particle_pos(out)
        return out

    def step(self):
        for s in range(self.steps):
            self.grid.deactivate_all()
            self.build_pid()
            self.p2g()
            self.grid_normalize()
            self.grid_boundary()
            for op in self.grid_postprocess:
                op()
            self.g2p()


def main():
    use_mciso = False

    ti.init(ti.gpu, make_block_local=False, kernel_profiler=True)
    mpm = MPMSolver([32] * 3)

    scene = tina.Scene(maxfaces=2**18, smoothing=True)
    if use_mciso:
        mciso = tina.MCISO(mpm.res)
        scene.add_object(mciso, tina.Classic())
    else:
        pars = tina.SimpleParticles(radius=0.01)
        scene.add_object(pars, tina.Classic())

    wire = tina.MeshToWire(tina.PrimitiveMesh.asset('cube'))
    scene.add_object(wire)

    vox = MeshVoxelizer([64] * 3)
    vox2 = MeshVoxelizer([64] * 3)
    verts, faces = tina.readobj('assets/bunny.obj', simple=True)
    vox.voxelize(verts[faces])
    verts, faces = tina.readobj('assets/cube.obj', simple=True)

    gui = ti.GUI()

    scene.init_control(gui, blendish=True)
    if use_mciso:
        w0 = gui.slider('w0', 0, 100, 0.1)
        w0.value = 11
        rad = gui.slider('rad', 0, 15, 1)
        rad.value = 4
        sig = gui.slider('sig', 0, 8, 0.1)
        sig.value = 5

    mpm.collide_volume(vox2, V(-.5, -1.35, -.5), V(.5, -.35, .5))
    #mpm.collide_sphere(V(0., -1., 0.), .5)

    def reset():
        mpm.seed_volume(vox, -.35, .35, 0., mpm.JELLY, 20)

    reset()
    while gui.running:
        scene.input(gui)
        if gui.is_pressed('r'):
            reset()
        if not gui.is_pressed(gui.SPACE):
            mpm.step()
        if use_mciso:
            mciso.march(mpm.x, w0.value, rad.value, sig.value)
        else:
            pars.set_particles(mpm.get_particle_pos())
            #colors = np.array(list(map(ti.hex_to_rgb, [0x068587, 0xED553B, 0xEEEEF0])))
            #pars.set_particle_colors(colors[mpm.material.to_numpy()])
        scene.render()
        gui.set_image(scene.img)
        gui.show()

    ti.kernel_profiler_print()

def main2():
    ti.init(ti.gpu, make_block_local=False)
    mpm = MPMSolver([128] * 2)

    gui = ti.GUI()
    while gui.running and not gui.get_event(gui.ESCAPE):
        if gui.is_pressed('r'):
            mpm.reset()
        if not gui.is_pressed(gui.SPACE):
            mpm.step()
        gui.circles(mpm.get_particle_pos() * 0.5 + 0.5)
        gui.show()

if __name__ == '__main__':
    main()
