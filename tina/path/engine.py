from ..advans import *


@ti.data_oriented
class PathEngine:
    def __init__(self, geom, lighting, mtltab, res=512):
        if isinstance(res, int): res = res, res
        self.res = ti.Vector(res)
        self.nrays = self.res.x * self.res.y

        self.img = ti.Vector.field(3, float, self.res)
        self.cnt = ti.field(int, self.res)

        self.geom = geom
        self.lighting = lighting
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
    def _get_image_e(self, out: ti.ext_arr(), notone: ti.template()):
        tonemap = ti.static((lambda x: x) if notone else aces_tonemap)
        self._f_get_image(out, tonemap, True)

    @ti.kernel
    def _get_image_f(self, out: ti.template()):
        self._f_get_image(out, lambda x: x, False)

    def get_image(self, out=None, notone=False):
        if out is None:
            out = np.zeros((*self.res, 3), dtype=np.float32)
            self._get_image_e(out, notone)
        else:
            self._get_image_f(out)
        return out

    @ti.func
    def trace_ray(self, I, maxdepth, surviverate):
        ro, rd = self.generate_ray(I)
        rc = V(1., 1., 1.)
        rl = V(0., 0., 0.)
        rs = 0.0

        rng = tina.TaichiRNG()
        for depth in range(maxdepth):
            ro, rd, rc, rl, rs = self.transmit_ray(ro, rd, rc, rl, rs, rng)
            rate = lerp(ti.tanh(Vavg(rc) * surviverate), 0.04, 0.95)
            if ti.random() >= rate:
                rc *= 0
            else:
                rc /= rate
            if not Vany(rc > 0):
                break

        return rl

    @ti.kernel
    def trace(self, maxdepth: int, surviverate: float):
        self.uniqid[None] += 1
        for i in ti.smart(self.stack):
            I = V(i // self.res.x, i % self.res.x)
            rl = self.trace_ray(I, maxdepth, surviverate)
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
    def hit_ray(self, ro, rd, rc, rng):
        near, ind, gid, uv = self.geom.hit(ro, rd)

        '''
        # travel volume
        vol_near, vol_far = tina.ray_aabb_hit(V3(-1.5), V3(1.5), ro, rd)
        vol_near = max(vol_near, 0)
        vol_far = min(vol_far, near)

        t = vol_near
        int_rho = 0.0
        ran = -ti.log(ti.random())
        step = 0.01
        t += step * ti.random()
        while t < vol_far:
            dt = min(step, vol_far - t)
            pos = ro + t * rd
            rho = 4.6 if (pos // 0.5).sum() % 2 == 0 else 0.0
            int_rho += rho * dt
            if ran < int_rho:
                new_rd = spherical(ti.random() * 2 - 1, ti.random())
                t += dt / 2
                ro += t * rd
                rd = new_rd
                rc *= 1.0
                gid = -2
                break
            t += dt
        '''

        return ro, rd, rc, near, ind, gid, uv

    @ti.func
    def transmit_ray(self, ro, rd, rc, rl, rs, rng):
        ro, rd, rc, near, ind, gid, uv = self.hit_ray(ro, rd, rc, rng)

        if gid == -1:
            # no hit
            rl += rc * self.lighting.background(rd)
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

            ro += nrm * eps * 8

            if rs < 1:
                rl += rc * (1 - rs) * material.emission()

            # cast shadow ray to lights
            rs = smoothstep(material.estimate_roughness(), 0.1, 0.5)
            if rs > 0:
                rl += rc * rs * self.shadow_ray(ro, rd, material, nrm, rng)

            # sample indirect light
            rd, ir_wei = material.sample(-rd, nrm, sign, rng)
            if rd.dot(nrm) < 0:
                # refract into / outof
                ro -= nrm * eps * 16

            tina.Input.clear_g_pars()

            rc *= ir_wei

        return ro, rd, rc, rl, rs

    @ti.func
    def shadow_ray(self, ro, rd, material, nrm, rng):
        ret = V(0., 0., 0.)
        li_rd, li_wei, li_dis = self.redirect_light(ro)
        li_wei *= max(0, li_rd.dot(nrm))
        if Vany(li_wei > 0):
            vol_ro, vol_rd, li_rc, near, ind, gid, uv = \
                    self.hit_ray(ro, li_rd, li_wei, rng)
            if gid == -1 or near > li_dis:
                # no shadow occlusion
                li_brdf = material.brdf(nrm, -rd, li_rd)
                ret = li_wei * li_brdf
        return ret

    @ti.func
    def redirect_light(self, ro):
        pos, ind, gid, wei = self.geom.sample_light_pos()

        mtlid = self.geom.get_material_id(ind, gid)
        material = self.mtltab.get(mtlid)
        color = material.emission()

        toli = pos - ro
        dis2 = toli.norm_sqr()
        toli = toli.normalized()
        fac = wei * color / dis2
        dis = ti.sqrt(dis2)
        return toli, fac, dis
