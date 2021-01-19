from ..advans import *
from .geometry import *


@ti.data_oriented
class BVHTree:
    def __init__(self, geom, N_tree=MAX, dim=3):
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
        print('[Tina] building tree...')
        self._build(data, pmin, pmax, np.arange(len(pmin)), 1)
        self._build_from_data(data.dir, data.min, data.max, data.ind)
        print('[Tina] building tree done')

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
    def hit(self, ro, rd):
        stack = tina.Stack.instance()
        near = inf
        ntimes = 0
        stack.clear()
        stack.push(1)
        hitind = -1
        hituv = V(0., 0.)
        while ntimes < self.N_tree and stack.size() != 0:
            curr = stack.pop()

            if self.dir[curr] == 0:
                ind = self.ind[curr]
                hit, depth, uv = self.geom.element_hit(ind, ro, rd)
                if hit != 0 and depth < near:
                    near = depth
                    hitind = ind
                    hituv = uv
                continue

            bmin, bmax = self.min[curr], self.max[curr]
            bnear, bfar = ray_aabb_hit(bmin, bmax, ro, rd)
            if bnear > bfar:
                continue

            ntimes += 1
            stack.push(curr * 2)
            stack.push(curr * 2 + 1)
        return near, hitind, hituv
