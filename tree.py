import taichi as ti
import numpy as np


pos = np.load('assets/fluid.npy') * 2 - 1


@ti.func
def ray_aabb_hit(bmin, bmax, ro, rd, inf=1e6, eps=1e-6):
    near = -inf
    far = inf
    hit = 1

    for i in ti.static(range(bmin.n)):
        if abs(rd[i]) < eps:
            if ro[i] < bmin[i] or ro[i] > bmax[i]:
                hit = 0
        else:
            i1 = (bmin[i] - ro[i]) / rd[i]
            i2 = (bmax[i] - ro[i]) / rd[i]

            far = min(far, max(i1, i2))
            near = max(near, min(i1, i2))

    if near > far:
        hit = 0

    return hit, near


@ti.func
def ray_triangle_hit(self, v0, v1, v2, ro, rd, inf=1e6, eps=1e-6):
    e1 = v1 - v0
    e2 = v2 - v0
    p = rd.cross(e2)
    det = e1.dot(p)
    s = ro - v0

    t, u, v = inf, 0.0, 0.0
    ipos = ti.Vector.zero(float, 3)
    inrm = ti.Vector.zero(float, 3)
    itex = ti.Vector.zero(float, 2)

    if det < 0:
        s = -s
        det = -det

    if det >= eps:
        u = s.dot(p)
        if 0 <= u <= det:
            q = s.cross(e1)
            v = rd.dot(q)
            if v >= 0 and u + v <= det:
                t = e2.dot(q)
                det = 1 / det
                t *= det
                u *= det
                v *= det
                inrm = e2.cross(e1).normalized()
                ipos = ro + t * rd
                itex = V(u, v)

    return t, ipos, inrm, itex


@ti.func
def ray_sphere_hit(pos, rad, ro, rd, inf=1e6, eps=1e-6):
    t = inf
    op = pos - ro
    b = op.dot(rd)
    det = b**2 - op.norm_sqr() + rad**2
    if det >= 0:
        det = ti.sqrt(det)
        t = b - det
        if t <= eps:
            t = b + det
            if t <= eps:
                t = inf
    return t < inf, t


@ti.data_oriented
class Stack:
    def __init__(self, N=64, L=512, field=None):
        self.val = ti.field(int) if field is None else field
        self.len = ti.field(int)
        ti.root.dense(ti.i, N).dynamic(ti.j, L).place(self.val)
        ti.root.dense(ti.i, N).place(self.len)

    def get(self, n):
        return self.Proxy(self, n)

    @ti.data_oriented
    class Proxy:
        def __init__(self, stack, n):
            self.stack = stack
            self.n = n

        def __getattr__(self, attr):
            return getattr(self.stack, attr)

        @ti.func
        def size(self):
            return self.len[self.n]

        @ti.func
        def clear(self):
            self.len[self.n] = 0

        @ti.func
        def push(self, val):
            l = self.len[self.n]
            self.val[self.n, l] = val
            self.len[self.n] = l + 1

        @ti.func
        def pop(self):
            l = self.len[self.n]
            val = self.val[self.n, l - 1]
            self.len[self.n] = l - 1
            return val


@ti.data_oriented
class BVHTree:
    def __init__(self, N_pars=1, N_tree=2**16, dim=2):
        self.N_tree = N_tree
        self.N_pars = N_pars
        self.dim = dim

        self.stack = Stack()

        self.dir = ti.field(int)
        self.min = ti.Vector.field(self.dim, float)
        self.max = ti.Vector.field(self.dim, float)
        self.ind = ti.field(int)
        self.tree = ti.root.pointer(ti.i, self.N_tree)
        self.tree.place(self.dir, self.min, self.max, self.ind)

        self.pos = ti.Vector.field(self.dim, float, self.N_pars)

    def build(self, pos):
        self.pos.from_numpy(pos)
        self._build(pos, np.arange(len(pos)), 1)

    def _build(self, pos, ind, curr):
        assert curr < self.N_tree, curr
        if not len(pos):
            return
        elif len(pos) <= 1:
            self.dir[curr] = 0
            self.ind[curr] = ind[0]
            return
        bmax = np.max(pos, axis=0)
        bmin = np.min(pos, axis=0)
        dir = np.argmax(bmax - bmin)
        sort = np.argsort(pos[:, dir])
        mid = len(sort) // 2
        l, r = pos[mid:], pos[:mid]
        li, ri = ind[mid:], ind[:mid]
        self.dir[curr] = 1 + dir
        self.min[curr] = bmin.tolist()
        self.max[curr] = bmax.tolist()
        self._build(l, li, curr * 2)
        self._build(r, ri, curr * 2 + 1)

    @ti.func
    def hit(self, stkid, ro, rd, inf=1e6):
        near = inf

        ntimes = 0
        stack = self.stack.get(stkid)
        stack.clear()
        stack.push(1)
        while ntimes < self.N_tree and stack.size() != 0:
            curr = stack.pop()

            if self.dir[curr] == 0:
                #print('zero', curr)
                ind = self.ind[curr]
                pos = self.pos[ind]
                hit, depth = ray_sphere_hit(pos, 0.002, ro, rd)
                if hit == 0:
                    continue
                if depth < near:
                    near = depth

            bmin, bmax = self.min[curr], self.max[curr]
            hit, depth = ray_aabb_hit(bmin, bmax, ro, rd)
            if hit == 0:
                #print('aabb', curr, bmin, bmax, ro, rd, depth)
                continue

            ntimes += 1
            stack.push(curr * 2)
            stack.push(curr * 2 + 1)

        print('ntimes', ntimes)
        return near


tree = BVHTree(N_pars=len(pos))
pos = pos[:, :tree.dim]
tree.build(pos)

@ti.kernel
def func(mx: float, my: float):
    ro = ti.Vector([-3.0, 0.0])
    rd = (ti.Vector([mx, my]) * 2 - 1 - ro).normalized()
    hit = tree.hit(0, ro, rd)
    print('hit', hit)

gui = ti.GUI()
while gui.running and not gui.get_event(gui.ESCAPE, gui.SPACE):
    func(*gui.get_cursor_pos())
    gui.circles(pos * 0.5 + 0.5, radius=512 * 0.002)
    gui.show()
