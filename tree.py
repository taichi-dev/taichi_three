import taichi as ti
import numpy as np
import taichi_inject

ti.init(ti.cuda)


inf = 1e6
eps = 1e-6


def rects(self, topleft, bottomright, radius=1, color=0xffff):
    topright = np.stack([topleft[:, 0], bottomright[:, 1]], axis=1)
    bottomleft = np.stack([bottomright[:, 0], topleft[:, 1]], axis=1)
    self.lines(topleft, topright, radius, color)
    self.lines(topright, bottomright, radius, color)
    self.lines(bottomright, bottomleft, radius, color)
    self.lines(bottomleft, topleft, radius, color)

ti.GUI.rects = rects
del rects


@ti.func
def ray_aabb_hit(bmin, bmax, ro, rd):
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
def ray_triangle_hit(self, v0, v1, v2, ro, rd):
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
def ray_sphere_hit(pos, rad, ro, rd):
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
class Stack:  # consumes 64 MiB by default:
    def __init__(self, N_mt=512**2, N_len=64, field=None):
        self.val = ti.field(int) if field is None else field
        self.blk1 = ti.root.dense(ti.i, N_mt)
        self.blk2 = self.blk1.dense(ti.j, N_len)
        self.blk2.place(self.val)

        self.len = ti.field(int, N_mt)

    def get(self, mtid):
        return self.Proxy(self, mtid)

    @ti.kernel
    def deactivate(self):
        for i in self.blk1:
            ti.deactivate(self.blk2, [i])

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
    def __init__(self, geom, N_tree=2**16, dim=3):
        self.geom = geom
        self.N_tree = N_tree
        self.dim = dim

        self.dir = ti.field(int)
        self.min = ti.Vector.field(self.dim, float)
        self.max = ti.Vector.field(self.dim, float)
        self.ind = ti.field(int)
        self.tree = ti.root.dense(ti.i, self.N_tree)
        self.tree.place(self.dir, self.min, self.max, self.ind)

    def build(self, pmin, pmax):
        assert len(pmin) == len(pmax)
        assert np.all(pmax >= pmin)
        data = lambda: None
        data.dir = self.dir.to_numpy()
        data.dir[:] = -1
        data.min = self.min.to_numpy()
        data.max = self.max.to_numpy()
        data.ind = self.ind.to_numpy()
        print('building tree...')
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
        assert self.dim == 2
        bmin = self.min.to_numpy()
        bmax = self.max.to_numpy()
        ind = self.active_indices()
        bmin, bmax = bmin[ind], bmax[ind]
        delta = bmax - bmin
        ind = np.any(delta >= 0.02, axis=1)
        bmin, bmax = bmin[ind], bmax[ind]
        bmin = bmin * 0.5 + 0.5
        bmax = bmax * 0.5 + 0.5
        gui.rects(bmin, bmax, color=0xff0000)

    @ti.func
    def hit(self, stack, ro, rd):
        near = inf
        ntimes = 0
        stack.clear()
        stack.push(1)
        hitind = -1
        while ntimes < self.N_tree and stack.size() != 0:
            curr = stack.pop()

            if self.dir[curr] == 0:
                ind = self.ind[curr]
                hit, depth = self.geom.hit(ind, ro, rd)
                if hit != 0 and depth < near:
                    near = depth
                    hitind = ind
                continue

            bmin, bmax = self.min[curr], self.max[curr]
            hit, depth = ray_aabb_hit(bmin, bmax, ro, rd)
            if hit == 0:
                continue

            ntimes += 1
            stack.push(curr * 2)
            stack.push(curr * 2 + 1)
        return near, hitind


@ti.data_oriented
class Camera:
    def __init__(self, scene, res=512):
        if isinstance(res, int): res = res, res
        self.res = ti.Vector(res)

        self.ro = ti.Vector.field(3, float)
        self.rd = ti.Vector.field(3, float)
        self.rc = ti.Vector.field(3, float)

        self.rays = ti.root.bitmasked(ti.ij, self.res)
        self.rays.place(self.ro)
        self.rays.place(self.rd)
        self.rays.place(self.rc)

        self.img = ti.Vector.field(3, float, self.res)
        self.cnt = ti.field(int, self.res)

        self.scene = scene
        self.stack = Stack()

    @ti.kernel
    def _get_image(self, out: ti.ext_arr()):
        for I in ti.grouped(self.img):
            val = self.img[I] / self.cnt[I]
            for k in ti.static(range(3)):
                out[I, k] = val[k]

    def get_image(self):
        img = np.zeros((*self.res, 3))
        self._get_image(img)
        return img

    @ti.kernel
    def deactivate(self):
        for I in ti.grouped(self.rays):
            ti.deactivate(self.rays, I)

    @ti.kernel
    def load_rays(self):
        for I in ti.grouped(ti.ndrange(*self.res)):
            bias = ti.Vector([ti.random(), ti.random()])
            uv = (I + bias) / self.res * 2 - 1
            ro = ti.Vector([0.0, 0.0, 3.0])
            rd = ti.Vector([uv.x, uv.y, -2.0]).normalized()
            rc = ti.Vector([1.0, 1.0, 1.0])
            self.ro[I] = ro
            self.rd[I] = rd
            self.rc[I] = rc

    @ti.func
    def transmit(self, near, ind,
            ro: ti.template(),
            rd: ti.template(),
            rc: ti.template()):
        if ind == -1:
            rc *= 0
            rd *= 0
        else:
            self.scene.geom.transmit(near, ind, ro, rd, rc)

    @ti.kernel
    def _step_rays(self):
        for I in ti.grouped(self.rays):
            stack = self.stack.get(I.y * self.res.x + I.x)
            ro = self.ro[I]
            rd = self.rd[I]
            rc = self.rc[I]
            if all(rd == 0):
                continue
            near, hitind = self.scene.hit(stack, ro, rd)
            self.transmit(near, hitind, ro, rd, rc)
            self.ro[I] = ro
            self.rd[I] = rd
            self.rc[I] = rc

    def step_rays(self):
        self._step_rays()
        self.stack.deactivate()

    @ti.kernel
    def update_image(self):
        for I in ti.grouped(ti.ndrange(*self.res)):
            rc = self.rc[I]
            self.img[I] += rc
            self.cnt[I] += 1


@ti.data_oriented
class Particles:
    def __init__(self, pos, rad=0.01, dim=3):
        self.pos = ti.Vector.field(dim, float, len(pos))
        self.rad = rad

        @ti.materialize_callback
        def init_pos():
            self.pos.from_numpy(pos)

    def build(self, tree):
        pos = self.pos.to_numpy()
        tree.build(pos - self.rad, pos + self.rad)

    @ti.func
    def hit(self, ind, ro, rd):
        pos = self.pos[ind]
        hit, depth = ray_sphere_hit(pos, self.rad, ro, rd)
        return hit, depth

    @ti.func
    def transmit(self, near, ind,
            ro: ti.template(),
            rd: ti.template(),
            rc: ti.template()):
        pos = ro + near * rd
        nrm = (pos - self.pos[ind]).normalized()

        rc *= max(0, -nrm.dot(rd))
        rd *= 0


pars = Particles(np.load('assets/fluid.npy') * 2 - 1, 0.01)
tree = BVHTree(geom=pars)
camera = Camera(scene=tree)

pars.build(tree)


gui = ti.GUI('BVH', tuple(camera.res.entries))
while gui.running and not gui.get_event(gui.ESCAPE, gui.SPACE):
    camera.deactivate()
    camera.load_rays()
    camera.step_rays()
    camera.update_image()
    gui.set_image(camera.get_image())
    gui.show()
