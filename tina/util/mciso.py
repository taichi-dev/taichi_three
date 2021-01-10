from ..common import *


@ti.data_oriented
class MCISO:
    def __init__(self, res, size=1):
        self.res = tovector(res)
        self.dim = len(res)
        self.dx = size / self.res.x

        from ._mciso_data import _et2, _et3
        et = [_et2, _et3][self.dim - 2]
        self.et = ti.Vector.field(self.dim, int, et.shape[:2])

        @ti.materialize_callback
        def init_tables():
            self.et.from_numpy(et)

        indices = [ti.ij, ti.ijk][self.dim - 2]

        self.m = ti.field(float)
        self.g = ti.Vector.field(self.dim, float)
        self.Jtab = ti.field(int)

        grid_size = 4096
        grid_block_size = 128
        leaf_block_size = 8
        self.offset = (-grid_size // 2,) * self.dim

        self.grid = ti.root.pointer(indices, grid_size // grid_block_size)
        block = self.grid.pointer(indices, grid_block_size // leaf_block_size)
        block.bitmasked(indices, leaf_block_size
                ).dense(ti.indices(self.dim), self.dim
                        ).place(self.Jtab, offset=self.offset + (0, ))
        block.dense(indices, leaf_block_size
                ).place(self.m, self.g, offset=self.offset)

        max_num_elems = 2**24
        self.face = ti.root.dynamic(ti.i, max_num_elems, max_num_elems // 128)
        self.vert = ti.root.dynamic(ti.i, max_num_elems, max_num_elems // 128)

        self.Js = ti.Vector.field(self.dim + 1, int)
        self.Jts = ti.Vector.field(self.dim, int)
        #self.Jis = ti.Vector.field(self.dim, int)
        self.face.dense(ti.j, self.dim).place(self.Js)
        self.face.place(self.Jts)
        #self.grid.dynamic(leaf_block_size * 32).place(self.Jis)

        self.vs = ti.Vector.field(self.dim, float)
        self.ns = ti.Vector.field(self.dim, float)
        self.vert.place(self.vs, self.ns)

        self.Js_n = ti.field(int, ())
        self.vs_n = ti.field(int, ())

        self.gwei = ti.field(float, 128)

    @ti.kernel
    def voxelize(self, pos: ti.template(), w0: float):
        for p in pos:
            Xp = pos[p] / self.dx
            base = ifloor(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for offset in ti.grouped(ti.ndrange(*(3,) * self.dim)):
                dpos = (offset - fx) * self.dx
                weight = float(w0)
                for i in ti.static(range(self.dim)):
                    weight *= list_subscript(w, offset[i])[i]
                self.m[base + offset] += weight

    @ti.kernel
    def sub_blur(self, radius: int, src: ti.template(), dst: ti.template(),
            axis: ti.template()):
        for I in ti.grouped(dst):
            dst[I] = src[I] * self.gwei[0]
        for I in ti.grouped(src):
            for i in range(1, radius + 1):
                dir = ti.Vector.unit(3, axis)
                wei = src[I] * self.gwei[i]
                dst[I + i * dir] += wei
                dst[I - i * dir] += wei

    @ti.kernel
    def init_gwei(self, radius: int, sigma: float):
        sum = -1.0
        for i in range(radius + 1):
            x = sigma * i / radius
            y = ti.exp(-x**2)
            self.gwei[i] = y
            sum += y * 2
        for i in range(radius + 1):
            self.gwei[i] /= sum

    def blur(self, radius, sigma):
        radius = int(radius)
        if radius <= 0: return
        self.init_gwei(radius, sigma)
        self.sub_blur(radius, self.m, self.g(0), 0)
        self.sub_blur(radius, self.g(0), self.g(1), 1)
        self.sub_blur(radius, self.g(1), self.m, 2)

    @ti.kernel
    def calc_norm(self):
        for I in ti.grouped(self.g):
            r = ti.Vector.zero(float, self.dim)
            for i in ti.static(range(self.dim)):
                d = ti.Vector.unit(self.dim, i, int)
                r[i] = self.m[I + d] - self.m[I - d]
            self.g[I] = -r.normalized(1e-5)

    @ti.kernel
    def gen_mesh(self):
        self.Js_n[None] = 0
        self.vs_n[None] = 0

        for I in ti.grouped(self.m):
            id = self.get_cubeid(I)
            for m in range(self.et.shape[1]):
                et = self.et[id, m]
                if et[0] == -1:
                    break

                Js_n = ti.atomic_add(self.Js_n[None], 1)
                for l in ti.static(range(self.dim)):
                    e = et[l]
                    J = ti.Vector(I.entries + [0])
                    if ti.static(self.dim == 2):
                        if e == 1 or e == 2: J.z = 1
                        if e == 2: J.x += 1
                        if e == 3: J.y += 1
                    else:
                        if e == 1 or e == 3 or e == 5 or e == 7: J.w = 1
                        elif e == 8 or e == 9 or e == 10 or e == 11: J.w = 2
                        if e == 1 or e == 5 or e == 9 or e == 10: J.x += 1
                        if e == 2 or e == 6 or e == 10 or e == 11: J.y += 1
                        if e == 4 or e == 5 or e == 6 or e == 7: J.z += 1
                    self.Js[Js_n, l] = J
                    self.Jtab[J] = 1

        for J in ti.grouped(self.Jtab):
            vs_n = ti.atomic_add(self.vs_n[None], 1)
            I = ti.Vector(ti.static(J.entries[:-1]))
            vs = I * 1.0
            ns = self.g[I]
            p1 = self.m[I]
            for t in ti.static(range(self.dim)):
                if J.entries[-1] == t:
                    K = I + ti.Vector.unit(self.dim, t, int)
                    p2 = self.m[K]
                    n2 = self.g[K]
                    p = (1 - p1) / (p2 - p1)
                    p = max(0, min(1, p))
                    vs[t] += p
                    ns += p * n2
            self.vs[vs_n] = (vs + 0.5) / self.res
            self.ns[vs_n] = ns.normalized(1e-4)
            self.Jtab[J] = vs_n

        for i in range(self.Js_n[None]):
            for l in ti.static(range(self.dim)):
                self.Jts[i][l] = self.Jtab[self.Js[i, l]]

    def march(self, pos, w0, rad, sig):
        self.grid.deactivate_all()
        self.voxelize(pos, w0)
        self.blur(rad, sig)
        self.calc_norm()
        self.gen_mesh()

    @ti.func
    def get_cubeid(self, I):
        id = 0
        if ti.static(self.dim == 2):
            i, j = I
            if self.m[i, j] > 1: id |= 1
            if self.m[i + 1, j] > 1: id |= 2
            if self.m[i, j + 1] > 1: id |= 4
            if self.m[i + 1, j + 1] > 1: id |= 8
        else:
            i, j, k = I
            if self.m[i, j, k] > 1: id |= 1
            if self.m[i + 1, j, k] > 1: id |= 2
            if self.m[i + 1, j + 1, k] > 1: id |= 4
            if self.m[i, j + 1, k] > 1: id |= 8
            if self.m[i, j, k + 1] > 1: id |= 16
            if self.m[i + 1, j, k + 1] > 1: id |= 32
            if self.m[i + 1, j + 1, k + 1] > 1: id |= 64
            if self.m[i, j + 1, k + 1] > 1: id |= 128
        return id

    @ti.func
    def pre_compute(self):
        pass

    @ti.func
    def get_nfaces(self):
        return self.Js_n[None]

    @ti.func
    def get_indiced_vert(self, i):
        return self.vs[i]

    @ti.func
    def get_indiced_norm(self, i):
        return self.ns[i]

    @ti.func
    def get_face_indices(self, n):
        i = self.Jts[n][2]
        j = self.Jts[n][1]
        k = self.Jts[n][0]
        return i, j, k

    @ti.func
    def get_face_verts(self, n):
        a = self.vs[self.Jts[n][2]]
        b = self.vs[self.Jts[n][1]]
        c = self.vs[self.Jts[n][0]]
        return a, b, c

    @ti.func
    def get_face_norms(self, n):
        a = self.ns[self.Jts[n][2]]
        b = self.ns[self.Jts[n][1]]
        c = self.ns[self.Jts[n][0]]
        return a, b, c

    @ti.func
    def get_transform(self):
        return ti.Matrix.identity(float, 4)


class _MCISO_Example1(MCISO):
    @ti.func
    def gauss(self, x):
        return ti.exp(-6 * x**2)

    @ti.kernel
    def touch(self, mx: float, my: float):
        for o in ti.grouped(ti.ndrange(*self.res)):
            p = o / self.res
            a = self.gauss((p - 0.5).norm() / 0.25)
            p.x -= mx - 0.5 / self.res.x
            p.y -= my - 0.5 / self.res.y
            if ti.static(self.dim == 3):
                p.z -= 0.5
            b = self.gauss(p.norm() / 0.25)
            r = max(a + b - 0.08, 0)
            if r <= 0:
                continue
            self.m[o] = r * 3

    def main(self):
        gui = ti.GUI('Marching cube')

        scene = tina.Scene(maxfaces=2**18, smoothing=True)
        scene.add_object(self)

        while gui.running and not gui.get_event(gui.ESCAPE):
            self.grid.deactivate_all()
            self.touch(*gui.get_cursor_pos())
            self.calc_norm()
            self.gen_mesh()

            scene.render()
            gui.set_image(scene.img)
            gui.show()


class _MCISO_Example2(MCISO):
    def main(self):
        gui = ti.GUI('Marching cube')

        scene = tina.Scene(maxfaces=2**20, smoothing=True)
        scene.add_object(self)

        pos = ti.Vector.field(3, float, 2**17)
        pos.from_numpy(np.random.rand(pos.shape[0], 3).astype(np.float32) * 0.6 - 0.3)

        w0 = gui.slider('w0', 0, 100, 0.1)
        w0.value = 42
        rad = gui.slider('rad', 0, 12, 1)
        rad.value = 3
        sig = gui.slider('sig', 0, 4, 0.1)
        sig.value = 1.2

        scene.init_control(gui, blendish=True)
        while gui.running:
            scene.input(gui)
            self.march(pos, w0.value, rad.value, sig.value)
            scene.render()
            gui.set_image(scene.img)
            gui.show()


if __name__ == '__main__':
    ti.init(ti.cuda, kernel_profiler=True)
    _MCISO_Example2([128] * 3).main()
    #_MCISO_Example1([128] * 3).main()
    ti.kernel_profiler_print()
