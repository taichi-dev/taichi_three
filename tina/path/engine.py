from ..advans import *


@ti.data_oriented
class PathEngine:
    def __init__(self, geom, lighting, mtltab, res=512):
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

        self.geom = geom
        self.lighting = lighting
        self.mtltab = mtltab
        self.stack = tina.Stack()

        self.W2V = ti.Matrix.field(4, 4, float, ())
        self.V2W = ti.Matrix.field(4, 4, float, ())

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
        for I in ti.grouped(ti.ndrange(*self.res)):
            bias = ti.Vector([ti.random(), ti.random()])
            uv = (I + bias) / self.res * 2 - 1
            # TODO: support customizing camera
            ro = mapply_pos(self.V2W[None], V(uv.x, uv.y, -1.0))
            ro1 = mapply_pos(self.V2W[None], V(uv.x, uv.y, +1.0))
            rd = (ro1 - ro).normalized()
            self.ro[I] = ro
            self.rd[I] = rd
            self.rc[I] = 1.0
            self.rl[I] = 0.0

    @ti.kernel
    def step_rays(self):
        for I in ti.smart(self.stack.ndrange(self.res)):
            ro = self.ro[I]
            rd = self.rd[I]
            rc = self.rc[I]
            rl = self.rl[I]
            if not Vall(rc < eps):
                ro, rd, rc, rl = self.transmit(ro, rd, rc, rl)
                self.ro[I] = ro
                self.rd[I] = rd
                self.rc[I] = rc
                self.rl[I] = rl

    @ti.kernel
    def update_image(self, strict: ti.template()):
        for I in ti.grouped(ti.ndrange(*self.res)):
            rc = self.rc[I]
            rl = self.rl[I]
            if strict and not Vall(rc < eps):
                continue
            self.img[I] += rl
            self.cnt[I] += 1

    def set_camera(self, view, proj):
        W2V = proj @ view
        V2W = np.linalg.inv(W2V)
        self.W2V.from_numpy(np.array(W2V, dtype=np.float32))
        self.V2W.from_numpy(np.array(V2W, dtype=np.float32))

    @ti.func
    def transmit(self, ro, rd, rc, rl):
        near, ind, gid, uv = self.geom.hit(ro, rd)
        if gid == -1:
            # no hit
            rl += rc * self.lighting.background(rd)
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
                'color': V(1., 1., 1.),
                'normal': nrm,
                'texcoord': tex,
            })

            mtlid = self.geom.get_material_id(ind, gid)
            material = self.mtltab.get(mtlid)

            ro += nrm * eps * 8
            li_clr = V(0., 0., 0.)
            if ti.static(0):
                for li_ind in range(self.lighting.get_nlights()):
                    # cast shadow ray to lights
                    new_rd, li_wei, li_dis = self.lighting.redirect(ro, li_ind)
                    li_wei *= max(0, new_rd.dot(nrm))
                    if Vall(li_wei <= 0):
                        continue
                    occ_near, occ_ind, occ_gid, occ_uv = self.geom.hit(ro, new_rd)
                    if occ_gid != -1 and occ_near < li_dis:  # shadow occlusion
                        continue  # but what if it's glass?
                    li_wei *= material.brdf(nrm, -rd, new_rd)
                    li_clr += li_wei

            # importance sampling indirect light by material
            mat_rd, mat_wei = material.sample(-rd, nrm, sign)

            # importance sampling indirect light by geometry
            geo_wei = V(0., 0., 0.)
            geo_rd, geo_wei = self.geom.choice(ro)
            geo_wei *= material.brdf(nrm, -rd, geo_rd)

            wei = V(0., 0., 0.)
            factor = lerp(max(eps, Vavg(geo_wei)) / max(eps, Vavg(geo_wei + mat_wei)), 0.06, 0.94)
            if ti.random() < factor:
                geo_rd = 0
                geo_wei = 0
                wei = geo_wei / factor
                rd = geo_rd
            else:
                wei = mat_wei / (1 - factor)
                rd = mat_rd

            if rd.dot(nrm) < 0:
                # refract into / outof
                ro -= nrm * eps * 16

            tina.Input.clear_g_pars()

            rl += rc * li_clr
            rc *= wei

        return ro, rd, rc, rl
