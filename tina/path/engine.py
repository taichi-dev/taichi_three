from ..advans import *
from ..util.stack import Stack


@ti.data_oriented
class PathEngine:
    def __init__(self, scene, res=512):
        if isinstance(res, int): res = res, res
        self.res = ti.Vector(res)

        self.ro = ti.Vector.field(3, float, self.res)
        self.rd = ti.Vector.field(3, float, self.res)
        self.rc = ti.Vector.field(3, float, self.res)

        #self.rays = ti.root.dense(ti.ij, self.res)
        #self.rays.place(self.ro)
        #self.rays.place(self.rd)
        #self.rays.place(self.rc)

        self.img = ti.Vector.field(3, float, self.res)
        self.cnt = ti.field(int, self.res)

        self.scene = scene
        self.stack = Stack()

    @ti.kernel
    def _get_image(self, out: ti.ext_arr()):
        for I in ti.grouped(self.img):
            val = self.img[I] / self.cnt[I]
            val = aces_tonemap(val)
            for k in ti.static(range(3)):
                out[I, k] = val[k]

    def get_image(self):
        img = np.zeros((*self.res, 3))
        self._get_image(img)
        return img

    @ti.kernel
    def load_rays(self):
        for I in ti.grouped(ti.ndrange(*self.res)):
            bias = ti.Vector([ti.random(), ti.random()])
            uv = (I + bias) / self.res * 2 - 1
            ro = ti.Vector([0.0, 0.0, -3.0])
            rd = ti.Vector([uv.x, uv.y, 2.0]).normalized()
            rc = ti.Vector([1.0, 1.0, 1.0])
            self.ro[I] = ro
            self.rd[I] = rd
            self.rc[I] = rc

    @ti.func
    def fallback(self, rd):
        t = 0.5 + rd.y + 0.5
        blue = ti.Vector([0.5, 0.7, 1.0])
        white = ti.Vector([1.0, 1.0, 1.0])
        ret = (1 - t) * white + t * blue
        ret = 0.1
        return ret

    @ti.func
    def transmit(self, near, ind, ro, rd, rc):
        if ind == -1:
            rc *= self.fallback(rd)
            rd *= 0
        else:
            ro, rd, rc = self.scene.geom.transmit(near, ind, ro, rd, rc)
        return ro, rd, rc

    @ti.kernel
    def step_rays(self):
        for I in ti.grouped(self.ro):
            stack = self.stack.get(I.y * self.res.x + I.x)
            ro = self.ro[I]
            rd = self.rd[I]
            rc = self.rc[I]
            if rd.norm_sqr() < 0.5:
                continue
            near, hitind = self.scene.hit(stack, ro, rd)
            ro, rd, rc = self.transmit(near, hitind, ro, rd, rc)
            self.ro[I] = ro
            self.rd[I] = rd
            self.rc[I] = rc

    @ti.kernel
    def update_image(self):
        for I in ti.grouped(ti.ndrange(*self.res)):
            rc = self.rc[I]
            ro = self.ro[I]
            rd = self.rd[I]
            if rd.norm_sqr() > 0.5:
                continue
            self.img[I] += rc
            self.cnt[I] += 1
