from ..advans import *
from ..util.stack import Stack


@ti.data_oriented
class PathEngine:
    def __init__(self, scene, lighting, res=512):
        if isinstance(res, int): res = res, res
        self.res = ti.Vector(res)

        self.ro = ti.Vector.field(3, float, self.res)
        self.rd = ti.Vector.field(3, float, self.res)
        self.rc = ti.Vector.field(3, float, self.res)
        self.rl = ti.Vector.field(3, float, self.res)

        #self.rays = ti.root.dense(ti.ij, self.res)
        #self.rays.place(self.ro)
        #self.rays.place(self.rd)
        #self.rays.place(self.rc)

        self.img = ti.Vector.field(3, float, self.res)
        self.cnt = ti.field(int, self.res)

        self.scene = scene
        self.lighting = lighting
        self.stack = Stack()

    @ti.kernel
    def _get_image(self, out: ti.ext_arr()):
        for I in ti.grouped(self.img):
            val = lerp((I // 8).sum() % 2, V(.4, .4, .4), V(.9, .9, .9))
            if self.cnt[I] != 0:
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
            rl = ti.Vector([0.0, 0.0, 0.0])
            self.ro[I] = ro
            self.rd[I] = rd
            self.rc[I] = rc
            self.rl[I] = rl

    @ti.func
    def background(self, rd):
        t = 0.5 + rd.y + 0.5
        blue = ti.Vector([0.5, 0.7, 1.0])
        white = ti.Vector([1.0, 1.0, 1.0])
        ret = (1 - t) * white + t * blue
        ret = 0.25
        return ret

    @ti.kernel
    def step_rays(self):
        for I in ti.grouped(self.ro):
            stack = self.stack.get(I.y * self.res.x + I.x)
            ro = self.ro[I]
            rd = self.rd[I]
            rc = self.rc[I]
            rl = self.rl[I]
            if (rc < eps).all():
                continue
            ro, rd, rc, rl = self.transmit(stack, ro, rd, rc, rl)
            self.ro[I] = ro
            self.rd[I] = rd
            self.rc[I] = rc
            self.rl[I] = rl

    @ti.kernel
    def update_image(self):
        for I in ti.grouped(ti.ndrange(*self.res)):
            rc = self.rc[I]
            rl = self.rl[I]
            if not (rc < eps).all():
                continue
            self.img[I] += rl
            self.cnt[I] += 1

    @ti.func
    def transmit(self, stack, ro, rd, rc, rl):
        near, ind, uv = self.scene.hit(stack, ro, rd)
        if ind == -1:
            # no hit
            near, ind, uv = self.lighting.hit(ro, rd)
            if ind == -1:
                # background
                rl += rc * self.background(rd)
            rc *= 0
        else:
            # hit object
            ro += near * rd
            nrm, tex = self.scene.geom.calc_geometry(near, ind, uv, ro, rd)
            if nrm.dot(rd) > 0:
                nrm = -nrm
            ro += nrm * eps * 8

            tina.Input.spec_g_pars({
                'pos': ro,
                'color': V(1., 1., 1.),
                'normal': nrm,
                'texcoord': tex,
            })

            li_clr = V(0., 0., 0.)
            for li_ind in range(self.lighting.get_nlights()):
                # cast shadow ray to lights
                new_rd, li_wei, li_dis = self.lighting.redirect(ro, li_ind)
                occ_near, _1, _2 = self.scene.hit(stack, ro, new_rd)
                if occ_near < li_dis:  # shadow occulsion
                    continue
                li_wei *= self.scene.geom.matr.brdf(nrm, rd, new_rd)
                li_clr += li_wei

            # sample indirect light
            rd, ir_wei = self.scene.geom.matr.sample(rd, nrm)

            tina.Input.clear_g_pars()

            rl += rc * li_clr
            rc *= ir_wei

        return ro, rd, rc, rl
