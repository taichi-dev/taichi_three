import taichi as ti
import numpy as np


def rects(self, topleft, bottomright, radius=1, color=0xffff):
    topright = np.stack([topleft[:, 0], bottomright[:, 1]], axis=1)
    bottomleft = np.stack([bottomright[:, 0], topleft[:, 1]], axis=1)
    self.lines(topleft, topright, radius, color)
    self.lines(topright, bottomright, radius, color)
    self.lines(bottomright, bottomleft, radius, color)
    self.lines(bottomleft, topleft, radius, color)

ti.GUI.rects = rects
del rects


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
    def __init__(self, N_mt=1024, N_len=4096, field=None):
        self.val = ti.field(int) if field is None else field
        self.len = ti.field(int)
        ti.root.dense(ti.i, N_mt).dynamic(ti.j, N_len).place(self.val)
        ti.root.dense(ti.i, N_mt).place(self.len)

    def get(self, mtid):
        return self.Proxy(self, mtid)

    @ti.data_oriented
    class Proxy:
        def __init__(self, stack, mtid):
            self.stack = stack
            self.mtid = mtid

        def __getattr__(self, attr):
            return getattr(self.stack, attr)

        @ti.func
        def size(self):
            return self.len[self.mtid]

        @ti.func
        def clear(self):
            self.len[self.mtid] = 0

        @ti.func
        def push(self, val):
            l = self.len[self.mtid]
            self.val[self.mtid, l] = val
            self.len[self.mtid] = l + 1

        @ti.func
        def pop(self):
            l = self.len[self.mtid]
            val = self.val[self.mtid, l - 1]
            self.len[self.mtid] = l - 1
            return val


@ti.data_oriented
class BVHTree:
    def __init__(self, N_pars, N_tree=2**16, dim=2):
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

    def build(self, pmin, pmax):
        print('building tree...')
        assert len(pmin) == len(pmax)
        assert np.all(pmax >= pmin)
        data = lambda: None
        data.dir = self.dir.to_numpy()
        data.dir[:] = -1
        data.min = self.min.to_numpy()
        data.max = self.max.to_numpy()
        data.ind = self.ind.to_numpy()
        self._build(data, pmin, pmax, np.arange(len(pmin)), 1)
        self._build_from_data(data.dir, data.min, data.max, data.ind)
        print('building tree done')

    @ti.kernel
    def _build_from_data(self,
            data_dir: ti.ext_arr(),
            data_min: ti.ext_arr(),
            data_max: ti.ext_arr(),
            data_ind: ti.ext_arr()):
        for i in range(self.dir.shape[0]):
            if data_dir[i] == -1:
                continue
            self.dir[i] = data_dir[i]
            for k in ti.static(range(self.dim)):
                self.min[i][k] = data_min[i, k]
                self.max[i][k] = data_max[i, k]
            self.ind[i] = data_ind[i]

    def _build(self, data, pmin, pmax, pind, curr):
        assert curr < self.N_tree, curr
        if not len(pind):
            return
        elif len(pind) <= 1:
            data.dir[curr] = 0
            data.ind[curr] = pind[0]
            data.min[curr] = pmin[0]
            data.max[curr] = pmax[0]
            return
        bmax = np.max(pmax, axis=0)
        bmin = np.min(pmin, axis=0)
        dir = np.argmax(bmax - bmin)
        sort = np.argsort(pmax[:, dir] + pmin[:, dir])
        mid = len(sort) // 2
        lsort = sort[:mid]
        rsort = sort[mid:]
        lmin, rmin = pmin[lsort], pmin[rsort]
        lmax, rmax = pmax[lsort], pmax[rsort]
        lind, rind = pind[lsort], pind[rsort]
        data.dir[curr] = 1 + dir
        data.ind[curr] = 0
        data.min[curr] = bmin
        data.max[curr] = bmax
        #print(lmax[:, dir].max(), rmin[:, dir].min())
        self._build(data, lmin, lmax, lind, curr * 2)
        self._build(data, rmin, rmax, rind, curr * 2 + 1)

    @ti.kernel
    def _active_indices(self, out: ti.ext_arr()):
        for curr in self.dir:
            if self.dir[curr] != 0:
                out[curr] = 1

    def active_indices(self):
        ind = np.zeros(self.N_tree, dtype=np.int32)
        self._active_indices(ind)
        return np.bool_(ind)

    def visualize(self, gui):
        bmin = self.min.to_numpy()
        bmax = self.max.to_numpy()
        ind = self.active_indices()
        bmin, bmax = bmin[ind], bmax[ind]
        delta = bmax - bmin
        ind = np.all(0.03 <= delta, axis=1)
        bmin, bmax = bmin[ind], bmax[ind]
        bmin = bmin * 0.5 + 0.5
        bmax = bmax * 0.5 + 0.5
        gui.rects(bmin, bmax, color=0xff0000)

    @ti.func
    def element_hit(self, ind, ro, rd):
        pos = self.pos[ind]
        hit, depth = ray_sphere_hit(pos, 0.002, ro, rd)
        return hit, depth

    @ti.func
    def hit(self, mtid, ro, rd, inf=1e6):
        near = inf

        ntimes = 0
        stack = self.stack.get(mtid)
        stack.clear()
        stack.push(1)
        while ntimes < self.N_tree and stack.size() != 0:
            curr = stack.pop()

            if self.dir[curr] == 0:
                ind = self.ind[curr]
                hit, depth = self.element_hit(ind, ro, rd)
                if hit != 0 and depth < near:
                    near = depth
                continue

            bmin, bmax = self.min[curr], self.max[curr]
            hit, depth = ray_aabb_hit(bmin, bmax, ro, rd)
            if hit == 0:
                continue

            ntimes += 1
            stack.push(curr * 2)
            stack.push(curr * 2 + 1)

        print('ntimes', ntimes)
        return near


tree = BVHTree(N_pars=len(pos))
pos = pos[:, :tree.dim]
tree.pos.from_numpy(pos)
tree.build(pos - 0.002, pos + 0.002)

@ti.kernel
def func(mx: float, my: float):
    ro = ti.Vector([-3.0, 0.0])
    rd = (ti.Vector([mx, my]) * 2 - 1 - ro).normalized()
    hit = tree.hit(0, ro, rd)
    print('hit', hit)

gui = ti.GUI()
while gui.running and not gui.get_event(gui.ESCAPE, gui.SPACE):
    func(*gui.get_cursor_pos())
    tree.visualize(gui)
    gui.circles(pos * 0.5 + 0.5, radius=512 * 0.002)
    gui.show()
