from tina.advans import *


@ti.func
def inside(p, a, b, c):
    u = (a - p).cross(b - p)
    v = (b - p).cross(c - p)
    w = (c - p).cross(a - p)
    ccw = u >= 0 and v >= 0 and w >= 0
    cw = u <= 0 and v <= 0 and w <= 0
    return ccw or cw


@ti.data_oriented
class MeshVoxelizer:
    def __init__(self, res):
        self.res = tovector(res)
        self.dx = 1 / self.res.x
        self.padding = 3

        self.voxels = ti.field(int, self.res)
        self.temp = ti.field(int, self.res)
        #self.block = ti.root.pointer(ti.ijk, self.res // 8)
        #self.block.dense(ti.ijk, 8).place(sel<F4>f.voxels)

    def voxelize(self, verts, vmin=None, vmax=None):
        if vmin is None or vmax is None:
            vmin, vmax = np.min(verts) - 0.1, np.max(verts) + 0.1
        verts = (verts - vmin) / (vmax - vmin)

        self._voxelize(verts)
        self._update(None)

        tmp = np.array(verts)
        tmp[..., (0, 1, 2)] = verts[..., (2, 0, 1)]
        self._voxelize(tmp)
        self._update(lambda x, y, z: (z, x, y))

        '''
        tmp = np.array(verts)
        tmp[..., (0, 1, 2)] = verts[..., (1, 2, 0)]
        self._voxelize(tmp)
        self._update(lambda x, y, z: (y, z, x))
        '''

        return vmin, vmax, verts

    @ti.kernel
    def _update(self, f: ti.template()):
        for I in ti.grouped(self.temp):
            if ti.static(f is None):
                self.voxels[I] = self.temp[I]
                self.temp[I] = 0
            else:
                J = V(*f(*I))
                self.voxels[I] = min(self.voxels[I], max(0, self.temp[J]))
                self.temp[J] = 0

    @ti.kernel
    def _voxelize(self, verts: ti.ext_arr()):
        for i in range(verts.shape[0]):
            jitter = V(-0.0576167239, -0.2560898629, 0.06716309129) * 1e-4
            a = V(verts[i, 0, 0], verts[i, 0, 1], verts[i, 0, 2]) + jitter
            b = V(verts[i, 1, 0], verts[i, 1, 1], verts[i, 1, 2]) + jitter
            c = V(verts[i, 2, 0], verts[i, 2, 1], verts[i, 2, 2]) + jitter

            bmin, bmax = min(a, b, c), max(a, b, c)

            pmin = max(self.padding, ifloor(bmin.xy / self.dx))
            pmax = min(self.res.xy - self.padding, ifloor(bmax.xy / self.dx) + 1)

            normal = (b - a).cross(c - a).normalized()

            if abs(normal.z) < 1e-10:
                continue

            for p in range(pmin.x, pmax.x):
                for q in range(pmin.y, pmax.y):
                    pos = (V(p, q) + 0.5) * self.dx
                    if inside(pos, a.xy, b.xy, c.xy):
                        base = V23(pos, 0.)
                        hei = int(-normal.dot(base - a) / normal.z / self.dx)
                        hei = min(hei, self.res.x - self.padding)
                        inc = 1 if normal.z > 0 else -1
                        for s in range(self.padding, hei):
                            self.temp[p, q, s] += inc


if __name__ == '__main__':
    ti.init(ti.cuda)

    vox = MeshVoxelizer([256] * 3)
    verts, faces = tina.readobj('assets/monkey.obj', simple=True)
    verts *= 0.5
    verts += 0.5

    scene = tina.Scene(taa=True)
    volume = tina.SimpleVolume(vox.res.x)
    scene.add_object(volume)
    #model = tina.MeshToWire(tina.MeshModel('assets/monkey.obj'))
    #scene.add_object(model)

    vox.voxelize(verts[faces])
    volume.set_volume_density(np.abs(vox.voxels.to_numpy()) * 0.05)

    gui = ti.GUI()
    while gui.running:
        scene.input(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.show()
