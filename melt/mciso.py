import taichi as ti
import numpy as np
import tina


@ti.data_oriented
class MCISO:
    def __init__(self, N, N_res=None, dim=3):
        self.N = N
        self.dim = dim
        self.N_res = N_res or N**self.dim

        from mciso_data import _et2, _et3
        et = [_et2, _et3][dim - 2]
        self.et = ti.Vector.field(dim, int, et.shape[:2])

        @ti.materialize_callback
        def init_tables():
            self.et.from_numpy(et)

        self.m = ti.field(float)
        self.g = ti.Vector.field(self.dim, float)
        self.Jtab = ti.field(int)
        indices = [ti.ij, ti.ijk][dim - 2]
        self.root = ti.root.dense(indices, 1)
        self.grid = self.root.pointer(indices, self.N // 16).pointer(indices, 16)
        self.grid.dense(ti.l, self.dim).place(self.Jtab)
        self.grid.place(self.m, self.g)

        self.Js = ti.Vector.field(self.dim + 1, int, (self.N_res, self.dim))
        self.Jts = ti.Vector.field(self.dim, int, self.N_res)
        self.vs = ti.Vector.field(self.dim, float, self.N_res)
        self.ns = ti.Vector.field(self.dim, float, self.N_res)
        self.Js_n = ti.field(int, ())
        self.vs_n = ti.field(int, ())

    @ti.kernel
    def march(self):
        self.Js_n[None] = 0
        self.vs_n[None] = 0

        for I in ti.grouped(self.g):
            r = ti.Vector.zero(float, self.dim)
            for i in ti.static(range(self.dim)):
                d = ti.Vector.unit(self.dim, i, int)
                r[i] = self.m[I + d] - self.m[I - d]
            self.g[I] = -r.normalized(1e-5)

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
            self.vs[vs_n] = vs
            self.ns[vs_n] = ns.normalized(1e-4)
            self.Jtab[J] = vs_n

        for i in range(self.Js_n[None]):
            for l in ti.static(range(self.dim)):
                self.Jts[i][l] = self.Jtab[self.Js[i, l]]

    def clear(self):
        self.root.deactivate_all()

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

    def get_mesh(self):
        ret = {}
        ret['f'] = self.Jts.to_numpy()[:self.Js_n[None]][:, ::-1]
        ret['v'] = (self.vs.to_numpy()[:self.vs_n[None]] + 0.5) / self.N
        ret['vn'] = self.ns.to_numpy()[:self.vs_n[None]]
        return ret

    @ti.func
    def pre_compute(self):
        pass

    @ti.func
    def get_nfaces(self):
        return self.Js_n[None]

    @ti.func
    def get_face_verts(self, n):
        a = (self.vs[self.Jts[n][2]] + 0.5) / self.N
        b = (self.vs[self.Jts[n][1]] + 0.5) / self.N
        c = (self.vs[self.Jts[n][0]] + 0.5) / self.N
        return a, b, c

    @ti.func
    def get_face_norms(self, n):
        a = self.ns[self.Jts[n][2]]
        b = self.ns[self.Jts[n][1]]
        c = self.ns[self.Jts[n][0]]
        return a, b, c

    @ti.func
    def sample_volume(self, pos):
        from ..advans import trilerp
        return trilerp(self.m, pos * self.N)

    @ti.func
    def get_transform(self):
        return ti.Matrix.identity(float, 4)


class _MCISO_Example(MCISO):
    @ti.func
    def gauss(self, x):
        return ti.exp(-6 * x**2)

    @ti.kernel
    def touch(self, mx: float, my: float):
        for o in ti.grouped(ti.ndrange(*[self.N] * self.dim)):
            p = o / self.N
            a = self.gauss((p - 0.5).norm() / 0.25)
            p.x -= mx - 0.5 / self.N
            p.y -= my - 0.5 / self.N
            if ti.static(self.dim == 3):
                p.z -= 0.5
            b = self.gauss(p.norm() / 0.25)
            r = max(a + b - 0.08, 0)
            if r <= 0:
                continue
            self.m[o] = r * 3

    def main(self):
        gui = ti.GUI('Marching cube')

        scene = tina.Scene()
        mesh = tina.SimpleMesh()
        scene.add_object(mesh)

        while gui.running and not gui.get_event(gui.ESCAPE):
            self.clear()
            self.touch(*gui.get_cursor_pos())
            self.march()

            Jts = self.Jts.to_numpy()[:self.Js_n[None], ::-1]
            vs = (self.vs.to_numpy()[:self.vs_n[None]] + 0.5) / self.N * 2 - 1
            ns = self.ns.to_numpy()[:self.vs_n[None]]
            mesh.set_face_verts(vs[Jts])
            mesh.set_face_norms(ns[Jts])
            scene.render()

            gui.set_image(scene.img)
            gui.text(f'Press space to save mesh to PLY ({len(vs)} verts, {len(Jts)} faces)', (0, 1))
            gui.show()

            if gui.is_pressed(gui.SPACE):
                writer = ti.PLYWriter(num_vertices=len(vs), num_faces=len(Jts))
                writer.add_vertex_pos(vs[:, 0], vs[:, 1], vs[:, 2])
                writer.add_faces(Jts)
                writer.export('mciso_output.ply')
                print('Mesh saved to mciso_output.ply')


if __name__ == '__main__':
    ti.init(ti.gpu)
    _MCISO_Example(64).main()
