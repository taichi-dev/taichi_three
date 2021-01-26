from ..advans import *
from ..core.shader import calc_viewdir


@ti.data_oriented
class SSR:
    def __init__(self, res, norm, coor, mtlid, mtltab, taa=False):
        self.res = tovector(res)
        self.img = ti.Vector.field(4, float, self.res)
        self.nsamples = ti.field(int, ())
        self.nsteps = ti.field(int, ())
        self.stepsize = ti.field(float, ())
        self.tolerance = ti.field(float, ())
        self.blurring = ti.field(int, ())
        self.norm = norm
        self.coor = coor
        self.mtlid = mtlid
        self.mtltab = mtltab
        self.taa = taa

        @ti.materialize_callback
        def init_params():
            self.nsamples[None] = 32 if not taa else 12
            self.nsteps[None] = 32 if not taa else 64
            self.stepsize[None] = 2
            self.tolerance[None] = 15
            self.blurring[None] = 4

    @ti.kernel
    def apply(self, image: ti.template()):
        for i, j in self.img:
            res = V(0., 0., 0., 0.)
            if ti.static(self.taa):
                res = self.img[i, j]
            else:
                rad = self.blurring[None]
                offs = rad // 2
                for k, l in ti.ndrange(rad, rad):
                    res += self.img[i + k - offs, j + l - offs]
                res /= rad**2
            image[i, j] *= 1 - res.w
            image[i, j] += res.xyz

    @ti.kernel
    def render(self, engine: ti.template(), image: ti.template()):
        for P in ti.grouped(image):
            if self.norm[P].norm_sqr() < eps:
                self.img[P] = 0
            else:
                self.render_at(engine, image, P)

    @ti.func
    def render_at(self, engine, image: ti.template(), P):
        normal = self.norm[P]
        texcoord = self.coor[P]
        mtlid = self.mtlid[P]

        p = P + engine.bias[None]
        vpos = V23(engine.from_viewport(p), engine.depth[P] / engine.maxdepth)
        pos = mapply_pos(engine.V2W[None], vpos)
        viewdir = calc_viewdir(engine, p)
        material = self.mtltab.get(mtlid)
        res = V(0., 0., 0., 0.)

        tina.Input.spec_g_pars({
                'pos': pos,
                'color': V(1., 1., 1.),
                'normal': normal,
                'texcoord': texcoord,
            })

        rng = tina.TaichiRNG()
        if ti.static(not self.taa):
            pid = P % self.blurring[None]
            rng = ti.static(tina.WangHashRNG(pid))

        nsamples = self.nsamples[None]
        nsteps = self.nsteps[None]
        for i in range(nsamples):
            odir, wei, rough = material.sample(viewdir, normal, 1, rng)

            step = self.stepsize[None] / (
                    ti.sqrt(1 - odir.dot(viewdir)**2) * nsteps)
            vtol = self.tolerance[None] * (
                    mapply_pos(engine.W2V[None], pos - viewdir / nsteps
                    ).z - mapply_pos(engine.W2V[None], pos).z)

            ro = pos + odir * rng.random() * step
            for j in range(nsteps):
                ro += odir * step
                vro = mapply_pos(engine.W2V[None], ro)
                if not all(-1 <= vro <= 1):
                    break
                D = engine.to_viewport(vro)
                depth = engine.depth[int(D)] / engine.maxdepth
                if vro.z - vtol < depth < vro.z:
                    clr = bilerp(image, D) * wei
                    res += V34(clr, 1.0)
                    break

        tina.Input.clear_g_pars()

        self.img[P] = res / nsamples
