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
    def _fast_export_image(self, out: ti.ext_arr(), blocksize: int):
        shape = self.res
        if blocksize != 0:
            shape //= blocksize
        for x, y in ti.ndrange(*shape):
            base = (y * shape.x + x) * 3
            I = V(x, y)
            val = lerp((I // 8).sum() % 2, V(.4, .4, .4), V(.9, .9, .9))
            if self.cnt[I] != 0:
                val = self.img[I] / self.cnt[I]
            r, g, b = val
            out[base + 0] = r
            out[base + 1] = g
            out[base + 2] = b

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

    @ti.kernel
    def trace(self, maxdepth: int, surviverate: float, blocksize: int):
        self.uniqid[None] += 1

        for i in ti.smart(self.stack):
            rng = tina.TaichiRNG()

            I = V(i % self.res.x, i // self.res.x)
            if blocksize != 0 and Vany(I % blocksize != 0):
                continue

            ro, rd = self.generate_ray(I, blocksize)
            rc = V(1., 1., 1.)
            rl = V(0., 0., 0.)
            rs = 0.0

            for depth in range(maxdepth):
                ro, rd, rc, rl, rs = self.transmit_ray(ro, rd, rc, rl, rs, rng)
                if not Vany(rc > 0):
                    break

            if blocksize != 0:
                I //= blocksize
            self.record_photon(I, rl)

    @ti.kernel
    def trace_light(self, maxdepth: int, surviverate: float):
        self.uniqid[None] += 1

        for i in ti.smart(self.stack):
            rng = tina.TaichiRNG()

            pos, ind, uv, gid, wei = self.geom.sample_light()
            if ind == -1:
                continue
            nrm, tex, mtlid = self.geom.calc_geometry(gid, ind, uv, pos)

            tina.Input.spec_g_pars({
                'pos': pos,
                'color': 1.,
                'normal': nrm,
                'texcoord': V(0., 0.),
            })

            material = self.mtltab.get(mtlid)
            color = material.emission()

            tina.Input.clear_g_pars()

            rd = tangentspace(nrm) @ spherical(ti.random() * 2 - 1, ti.random())
            rc = V3(wei * color)
            ro = pos + rd * eps * 8
            rn = 0

            for depth in range(maxdepth):
                ro, rd, rc, rn = self.transmit_light(ro, rd, rc, rn, rng)
                if not Vany(rc > 0):
                    break

    @ti.func
    def record_photon(self, I, rl):
        self.img[I] += rl
        self.cnt[I] += 1

    @ti.func
    def generate_ray(self, I, blocksize):
        bias = V(.5, .5) * blocksize
        if blocksize == 0:
            bias = V(ti.random(), ti.random())
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
    def transmit_light(self, ro, rd, rc, rn, rng):
        near, ind, gid, uv = self.geom.hit(ro, rd)

        if gid == -1:
            # no hit
            rc *= 0

        elif gid != -2:
            rn += near

            # hit object
            ro += near * rd
            nrm, tex, mtlid = self.geom.calc_geometry(gid, ind, uv, ro)

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

            material = self.mtltab.get(mtlid)

            # sample indirect light
            new_rd, ir_wei, rs = material.sample(-rd, nrm, sign, rng)
            if new_rd.dot(nrm) < 0:
                # refract into / outof
                ro -= nrm * eps * 8
            else:
                ro += nrm * eps * 8

            # cast shadow ray to lights
            self.shadow_light(ro, rd, rc, rn, material, nrm, rng)

            tina.Input.clear_g_pars()

            rd = new_rd
            rc *= ir_wei

        return ro, rd, rc, rn

    @ti.func
    def shadow_light(self, ro, rd, rc, rn, material, nrm, rng):
        vpos = mapply_pos(self.W2V[None], ro)
        if all(-1 < vpos <= 1):
            vpos.z = -1.0
            ro0 = mapply_pos(self.V2W[None], vpos)
            new_rd = (ro0 - ro).normalized()
            li_dis = (ro0 - ro).norm()
            li_clr = rc * max(0, -rd.dot(nrm)) / rn**2
            if Vany(li_clr > 0):
                occ_near, occ_ind, occ_gid, occ_uv = self.geom.hit(ro, new_rd)
                if occ_gid == -1 or occ_near >= li_dis:  # no shadow occlusion
                    li_clr *= material.brdf(nrm, -rd, new_rd)
                    I = ifloor((vpos.xy * 0.5 + 0.5) * self.res)
                    self.record_photon(I, li_clr)

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
            nrm, tex, mtlid = self.geom.calc_geometry(gid, ind, uv, ro)
            material = self.mtltab.get(mtlid)

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

            if rs < 1:
                rl += rc * (1 - rs) * material.emission()

            # sample indirect light
            new_rd, ir_wei, brdf_pdf = material.sample(-rd, nrm, sign, rng)
            if new_rd.dot(nrm) < 0:
                # refract into / outof
                ro -= nrm * eps * 8
            else:
                ro += nrm * eps * 8

            # cast shadow ray to lights
            li_rd, li_wei, li_pdf = self.redirect_light(ro)

            li_wei *= max(0, nrm.dot(li_rd))
            rs = li_pdf**2 / (li_pdf**2 + brdf_pdf**2)

            li_brdf = material.brdf(nrm, -rd, li_rd)
            rl += rc * rs * li_brdf * li_wei

            tina.Input.clear_g_pars()

            rd = new_rd
            rc *= ir_wei

        return ro, rd, rc, rl, rs

    @ti.func
    def redirect_light(self, ro):
        pos, ind, uv, gid, wei = self.geom.sample_light()

        toli, fac, dis, pdf = V3(0.), V3(0.), inf, 0.
        if ind != -1:
            nrm, tex, mtlid = self.geom.calc_geometry(gid, ind, uv, pos)

            tina.Input.spec_g_pars({
                'pos': pos,
                'color': 1.,
                'normal': nrm,
                'texcoord': tex,
            })

            material = self.mtltab.get(mtlid)
            color = material.emission()

            tina.Input.clear_g_pars()

            toli = pos - ro
            dis2 = toli.norm_sqr()
            toli = toli.normalized()
            wei *= abs(toli.dot(nrm))
            pdf = dis2 / wei
            fac = color * wei / (dis2 + eps)
            dis = ti.sqrt(dis2)

            if Vany(fac > 0):
                near, ind, gid, uv = self.geom.hit(ro, toli)
                if gid != -1 and near < dis:
                    # shadow occlusion
                    fac *= 0

        return toli, fac, pdf

    @ti.func
    def background(self, rd):
        if ti.static(hasattr(self, 'skybox')):
            return self.skybox.sample(rd)
        else:
            return 0.0
