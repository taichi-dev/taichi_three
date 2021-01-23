from ..advans import *


@ti.data_oriented
class BidirEngine:
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
    def trace_ray(self, I, rw, maxdepth, surviverate):
        ro, rd, rc, rl = self.generate_ray(I)
        ray_ro = ro
        ray_rc = rc
        ray_depth = 0

        rng = tina.TaichiRNG()
        for depth in range(maxdepth):
            ray_ro = ro
            ray_rc = rc
            ray_depth = depth

            ro, rd, rc, rl, rw = self.transmit_ray(ro, rd, rc, rl, rw, rng)
            rate = lerp(ti.tanh(Vavg(rc) * surviverate), 0.04, 0.95)
            if ti.random() >= rate:
                rc *= 0
            else:
                rc /= rate
            if not Vany(rc > 0):
                break

        return ray_ro, ray_rc, ray_depth, rl

    @ti.kernel
    def trace(self, maxdepth: int, surviverate: float):
        self.uniqid[None] += 1
        for i in ti.smart(self.stack):
            I = V(i // self.res.x, i % self.res.x)
            rw = tina.random_wav(self.uniqid[None] + I.y)
            ray_ro, ray_rc, ray_depth = self.trace_ray(I, rw, maxdepth, surviverate)
            lay_ro, lay_rc, lay_depth = self.trace_lay(rw, maxdepth, surviverate)

            self.img[I] += rl * tina.wav_to_rgb(rw)
            self.cnt[I] += 1

    @ti.func
    def trace_lay(self, rw, maxdepth, surviverate):
        ro, rd, rc = self.generate_lay()
        lay_ro = ro
        lay_rc = rc
        lay_depth = 0

        rng = tina.TaichiRNG()
        for depth in range(maxdepth):
            lay_ro = ro
            lay_rc = rc
            lay_depth = depth

            ro, rd, rc, rw = self.transmit_lay(ro, rd, rc, rw, rng)
            rate = lerp(ti.tanh(Vavg(rc) * surviverate), 0.04, 0.95)
            if ti.random() >= rate:
                rc *= 0
            else:
                rc /= rate
            if not Vany(rc > 0):
                break

        return lay_ro, lay_rc, lay_depth

    @ti.func
    def generate_ray(self, I):
        bias = ti.Vector([ti.random(), ti.random()])
        uv = (I + bias) / self.res * 2 - 1
        ro = mapply_pos(self.V2W[None], V(uv.x, uv.y, -1.0))
        ro1 = mapply_pos(self.V2W[None], V(uv.x, uv.y, +1.0))
        rd = (ro1 - ro).normalized()
        rc = 1.0
        rl = 0.0
        return ro, rd, rc, rl

    @ti.func
    def generate_lay(self):
        ind = ti.random(int) % self.lighting.get_nlights()
        ro, rd = self.lighting.emit_light(ind)
        rc = 1.0
        return ro, rd, rc

    @ti.func
    def update_image_light(self, uv, rc, rw):
        I = ifloor((uv * 0.5 + 0.5) * self.res)
        self.img[I] += rc * tina.wav_to_rgb(rw)
        self.cnt[I] += 1

    def set_camera(self, view, proj):
        W2V = proj @ view
        V2W = np.linalg.inv(W2V)
        self.W2V.from_numpy(np.array(W2V, dtype=np.float32))
        self.V2W.from_numpy(np.array(V2W, dtype=np.float32))

    @ti.func
    def transmit_ray(self, ro, rd, rc, rl, rw, rng):
        near, ind, gid, uv = self.geom.hit(ro, rd)

        # travel volume
        vol_near, vol_far = tina.ray_aabb_hit(V3(-1.5), V3(1.5), ro, rd)
        vol_near = max(vol_near, 0)
        vol_far = min(vol_far, near)

        t = vol_near
        vol_hit = 0
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
                #rc *= tina.rgb_at_wav(V(1., 0., 0.), rw)
                vol_hit = 1
                break
            t += dt

        if vol_hit == 0:
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

    @ti.func
    def transmit_lay(self, ro, rd, rc, rw, rng):
        near, ind, gid, uv = self.geom.hit(ro, rd)

        # travel volume
        vol_near, vol_far = tina.ray_aabb_hit(V3(-1.5), V3(1.5), ro, rd)
        vol_near = max(vol_near, 0)
        vol_far = min(vol_far, near)

        t = vol_near
        vol_hit = 0
        while t < vol_far:
            dt = min(0.05 * ti.random(), vol_far - t)
            rho = 0.6
            if ti.random() >= ti.exp(-dt * rho):
                new_rd = spherical(ti.random() * 2 - 1, ti.random())
                t += dt / 2
                ro += t * rd
                rd = new_rd
                vol_hit = 1
                break
            t += dt

        if vol_hit == 0:
            if gid == -1:
                # no hit
                rc *= 0
            else:
                if vol_hit == 0:
                    # hit object
                    ro += near * rd
                    nrm, tex = self.geom.calc_geometry(near, gid, ind, uv, ro, rd)

                    sign = 1
                    if nrm.dot(rd) > 0:
                        sign = -1
                        nrm = -nrm

                    tina.Input.spec_g_pars({
                        'pos': ro,
                        'color': 1.0,
                        'normal': nrm,
                        'texcoord': tex,
                    })

                    mtlid = self.geom.get_material_id(ind, gid)
                    material = self.mtltab.get(mtlid)

                    ro += nrm * eps * 8

                    # cast shadow ray to camera
                    vpos = mapply_pos(self.W2V[None], ro)
                    if all(-1 < vpos <= 1):
                        vpos.z = -1.0
                        ro0 = mapply_pos(self.V2W[None], vpos)
                        new_rd = (ro0 - ro).normalized()
                        li_dis = (ro0 - ro).norm()
                        li_clr = rc * max(0, -rd.dot(nrm))
                        if Vany(li_clr > 0):
                            occ_near, occ_ind, occ_gid, occ_uv = self.geom.hit(ro, new_rd)
                            if occ_gid == -1 or occ_near >= li_dis:  # no shadow occlusion
                                li_clr *= material.wav_brdf(nrm, -rd, new_rd, rw)
                                self.update_image_light(vpos.xy, li_clr, rw)

                    # sample indirect light
                    rd, ir_wei = material.wav_sample(-rd, nrm, sign, rng, rw)
                    if rd.dot(nrm) < 0:
                        # refract into / outof
                        ro -= nrm * eps * 16

                    tina.Input.clear_g_pars()

                    rc *= ir_wei

        return ro, rd, rc, rw
