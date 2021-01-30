from ..advans import *


@ti.data_oriented
class PathEngine:
    def __init__(self, geom, mtltab, res=512):
        if isinstance(res, int): res = res, res
        self.res = ti.Vector(res)
        self.nrays = self.res.x * self.res.y

        self.img = ti.Vector.field(3, float, self.res)
        self.cnt = ti.field(int, self.res)

        self.geom = geom
        self.mtltab = mtltab
        self.stack = tina.Stack(N_mt=self.nrays)

        self.W2V = ti.Matrix.field(4, 4, float, ())
        self.V2W = ti.Matrix.field(4, 4, float, ())
        self.uniqid = ti.field(int, ())

        @ti.materialize_callback
        @ti.kernel
        def init_engine():
            self.W2V[None] = ti.Matrix.identity(float, 4)
            self.W2V[None][2, 2] = -1
            self.V2W[None] = ti.Matrix.identity(float, 4)
            self.V2W[None][2, 2] = -1

    def clear_image(self):
        self.img.fill(0)
        self.cnt.fill(0)

    @ti.kernel
    def _get_image(self, out: ti.ext_arr(), raw: ti.template()):
        for I in ti.grouped(self.img):
            val = lerp((I // 8).sum() % 2, V(.4, .4, .4), V(.9, .9, .9))
            if self.cnt[I] != 0:
                val = self.img[I] / self.cnt[I]
                out[I, 3] = 1.
            else:
                out[I, 3] = 0.
            if not all(val >= 0 or val <= 0):
                val = V(.9, .4, .9)
            else:
                if ti.static(not raw):
                    val = aces_tonemap(val)
            for k in ti.static(range(3)):
                out[I, k] = val[k]

    def get_image(self, raw=False):
        out = np.zeros((*self.res, 4), dtype=np.float32)
        self._get_image(out, raw)
        return out

    @ti.func
    def trace_ray(self, I, maxdepth, surviverate, rng):
        ro, rd = self.generate_ray(I)
        rc = V(1., 1., 1.)
        rl = V(0., 0., 0.)
        rs = 0.0

        for depth in range(maxdepth):
            ro, rd, rc, rl, rs = self.transmit_ray(ro, rd, rc, rl, rs, rng)
            #rate = lerp(ti.tanh(Vavg(rc) * surviverate), 0.06, 0.996)
            #if ti.random() >= rate:
            #    rc *= 0
            #else:
            #    rc /= rate
            if not Vany(rc > 0):
                break

        return rl

    @ti.kernel
    def trace(self, maxdepth: int, surviverate: float):
        self.uniqid[None] += 1

        for i in ti.smart(self.stack):
            rng = tina.TaichiRNG()
            #rng = tina.WangHashRNG(V(i, self.uniqid[None]))
            #rng = tina.HammersleyRNG(i, self.uniqid[None])
            #rng = tina.SobolRNG(self.sobol, i)

            I = V(i % self.res.x, i // self.res.x)
            rl = self.trace_ray(I, maxdepth, surviverate, rng)
            self.img[I] += rl
            self.cnt[I] += 1

    @ti.func
    def generate_ray(self, I):
        bias = ti.Vector([ti.random(), ti.random()])
        uv = (I + bias) / self.res * 2 - 1
        ro = mapply_pos(self.V2W[None], V(uv.x, uv.y, -1.0))
        ro1 = mapply_pos(self.V2W[None], V(uv.x, uv.y, +1.0))
        rd = (ro1 - ro).normalized()
        return ro, rd

    def set_camera(self, view, proj):
        W2V = proj @ view
        V2W = np.linalg.inv(W2V)
        self.W2V.from_numpy(np.array(W2V, dtype=np.float32))
        self.V2W.from_numpy(np.array(V2W, dtype=np.float32))

    @ti.func
    def transmit_ray(self, ro, rd, rc, rl, rs, rng):
        near, ind, gid, uv = self.geom.hit(ro, rd)

        if gid == -1:
            # no hit
            rl += rc * self.background(rd)
            rc *= 0

        elif gid != -2:
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

            if rs < 1:
                rl += rc * (1 - rs) * material.emission()

            # sample indirect light
            new_rd, ir_wei, rs = material.sample(-rd, nrm, sign, rng)
            if new_rd.dot(nrm) < 0:
                # refract into / outof
                ro -= nrm * eps * 8
            else:
                ro += nrm * eps * 8

            # cast shadow ray to lights
            if rs > 0:
                rl += rc * rs * self.shadow_ray(ro, rd, material, nrm, rng)

            tina.Input.clear_g_pars()

            rd = new_rd
            rc *= ir_wei

        return ro, rd, rc, rl, rs

    @ti.func
    def shadow_ray(self, ro, rd, material, nrm, rng):
        ret = V(0., 0., 0.)
        li_rd, li_wei, li_dis = self.redirect_light(ro)
        li_wei *= max(0, li_rd.dot(nrm))
        if Vany(li_wei > 0):
            near, ind, gid, uv = self.geom.hit(ro, li_rd)
            if gid == -1 or near > li_dis:
                # no shadow occlusion
                li_brdf = material.brdf(nrm, -rd, li_rd)
                ret = li_wei * li_brdf
        return ret

    @ti.func
    def redirect_light(self, ro):
        pos, ind, gid, wei = self.geom.sample_light_pos(ro)
        toli, fac, dis = V3(0.), V3(0.), inf
        if ind != -1:
            mtlid = self.geom.get_material_id(ind, gid)
            material = self.mtltab.get(mtlid)
            color = material.emission()

            toli = pos - ro
            dis2 = toli.norm_sqr()
            toli = toli.normalized()
            fac = wei * color / (dis2 + eps)
            dis = ti.sqrt(dis2)
        return toli, fac, dis

    @ti.func
    def background(self, rd):
        if ti.static(hasattr(self, 'skybox')):
            return self.skybox.sample(rd)
        else:
            return 0.0
