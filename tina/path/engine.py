from ..advans import *


@ti.data_oriented
class PathEngine:
    def __init__(self, geom, lighting, mtltab, res=512):
        if isinstance(res, int): res = res, res
        self.res = ti.Vector(res)
        self.nrays = self.res.x * self.res.y

        self.ro = ti.Vector.field(3, float, self.nrays)
        self.rd = ti.Vector.field(3, float, self.nrays)
        self.rc = ti.field(float, self.nrays)
        self.rl = ti.field(float, self.nrays)
        self.rw = ti.field(float, self.nrays)
        self.rI = ti.Vector.field(2, int, self.nrays)

        self.img = ti.Vector.field(3, float, self.res)
        self.cnt = ti.field(int, self.res)

        self.geom = geom
        self.lighting = lighting
        self.mtltab = mtltab
        self.stack = tina.Stack(N_mt=self.nrays)

        self.W2V = ti.Matrix.field(4, 4, float, ())
        self.V2W = ti.Matrix.field(4, 4, float, ())
        self.surate = ti.field(float, ())
        self.uniqid = ti.field(int, ())

        @ti.materialize_callback
        @ti.kernel
        def init_engine():
            self.surate[None] = 2
            self.W2V[None] = ti.Matrix.identity(float, 4)
            self.W2V[None][2, 2] = -1
            self.V2W[None] = ti.Matrix.identity(float, 4)
            self.V2W[None][2, 2] = -1

    def clear_image(self):
        self.img.fill(0)
        self.cnt.fill(0)

    @ti.func
    def _f_get_image(self, out: ti.template(),
                     tonemap: ti.template(), is_ext: ti.template()):
        for I in ti.grouped(self.img):
            val = lerp((I // 8).sum() % 2, V(.4, .4, .4), V(.9, .9, .9))
            if self.cnt[I] != 0:
                val = self.img[I] / self.cnt[I]
            if not all(val >= 0 or val <= 0):  # NaN?
                val = V(.9, .4, .9)
            val = tonemap(val)
            if ti.static(is_ext):
                for k in ti.static(range(3)):
                    out[I, k] = val[k]
            else:
                out[I] = val

    @ti.kernel
    def _get_image_e(self, out: ti.ext_arr()):
        self._f_get_image(out, aces_tonemap, True)

    @ti.kernel
    def _get_image_f(self, out: ti.template()):
        self._f_get_image(out, lambda x: x, False)

    def get_image(self, out=None):
        if out is None:
            out = np.zeros((*self.res, 3), dtype=np.float32)
            self._get_image_e(out)
        else:
            self._get_image_f(out)
        return out

    @ti.kernel
    def load_rays(self):
        self.uniqid[None] += 1
        for I in ti.grouped(ti.ndrange(*self.res)):
            i = I.dot(V(1, self.res.x))
            bias = ti.Vector([ti.random(), ti.random()])
            uv = (I + bias) / self.res * 2 - 1
            ro = mapply_pos(self.V2W[None], V(uv.x, uv.y, -1.0))
            ro1 = mapply_pos(self.V2W[None], V(uv.x, uv.y, +1.0))
            rd = (ro1 - ro).normalized()
            rw = tina.random_wav(self.uniqid[None] + I.y)
            self.ro[i] = ro
            self.rd[i] = rd
            self.rc[i] = 1.0
            self.rl[i] = 0.0
            self.rw[i] = rw
            self.rI[i] = I

    @ti.func
    def ray_alive(self, i):
        return Vany(self.rc[i] > 0)

    @ti.kernel
    def kill_rays(self):
        for i in self.ro:
            rc = self.rc[i]
            if not self.ray_alive(i):
                continue
            rate = lerp(ti.tanh(Vavg(rc) * self.surate[None]), 0.04, 0.95)
            if ti.random() >= rate:
                self.rc[i] = 0.0
            else:
                self.rc[i] = rc / rate

    @ti.kernel
    def count_rays(self) -> int:
        count = 0
        for i in self.ro:
            rc = self.rc[i]
            if self.ray_alive(i):
                count += 1
        return count

    @ti.kernel
    def step_rays(self):
        for i in ti.smart(self.stack):
            if self.ray_alive(i):
                rng = tina.TaichiRNG()
                ro = self.ro[i]
                rd = self.rd[i]
                rc = self.rc[i]
                rl = self.rl[i]
                rw = self.rw[i]
                ro, rd, rc, rl, rw = self.transmit(ro, rd, rc, rl, rw, rng)
                self.ro[i] = ro
                self.rd[i] = rd
                self.rc[i] = rc
                self.rl[i] = rl
                self.rw[i] = rw

    @ti.kernel
    def update_image(self, strict: ti.template()):
        for i in self.ro:
            if strict and self.ray_alive(i):
                continue
            I = self.rI[i]
            rc = self.rc[i]
            rl = self.rl[i]
            rw = self.rw[i]
            self.img[I] += rl * tina.wav_to_rgb(rw)
            self.cnt[I] += 1

    def set_camera(self, view, proj):
        W2V = proj @ view
        V2W = np.linalg.inv(W2V)
        self.W2V.from_numpy(np.array(W2V, dtype=np.float32))
        self.V2W.from_numpy(np.array(V2W, dtype=np.float32))

    @ti.func
    def transmit(self, ro, rd, rc, rl, rw, rng):
        near, ind, gid, uv = self.geom.hit(ro, rd)
        if gid == -1:
            # no hit
            rl += rc * self.lighting.background(rd, rw)
            rc *= 0
        else:
            # hit object
            ro += near * rd
            nrm, tex = self.geom.calc_geometry(near, gid, ind, uv, ro, rd)

            sign = 1
            if nrm.dot(rd) > 0:
                sign = -1
                nrm = -nrm

            tina.Input.spec_g_pars({
                'pos': ro,
                'color': 1.,
                'normal': nrm,
                'texcoord': tex,
            })

            mtlid = self.geom.get_material_id(ind, gid)
            material = self.mtltab.get(mtlid)

            ro += nrm * eps * 8

            li_clr = 0.
            for li_ind in range(self.lighting.get_nlights()):
                # cast shadow ray to lights
                new_rd, li_wei, li_dis = self.lighting.redirect(ro, li_ind, rw)
                li_wei *= max(0, new_rd.dot(nrm))
                if Vall(li_wei <= 0):
                    continue
                occ_near, occ_ind, occ_gid, occ_uv = self.geom.hit(ro, new_rd)
                if occ_gid != -1 and occ_near < li_dis:  # shadow occlusion
                    continue  # but what if it's glass?
                li_wei *= material.wav_brdf(nrm, -rd, new_rd, rw)
                li_clr += li_wei

            # sample indirect light
            rd, ir_wei = material.wav_sample(-rd, nrm, sign, rng, rw)
            if rd.dot(nrm) < 0:
                # refract into / outof
                ro -= nrm * eps * 16

            tina.Input.clear_g_pars()

            rl += rc * li_clr
            rc *= ir_wei

        return ro, rd, rc, rl, rw
