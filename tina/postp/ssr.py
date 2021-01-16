from ..advans import *
from ..core.shader import calc_viewdir


@ti.data_oriented
class SSR:
    def __init__(self, res, norm):
        self.res = tovector(res)
        self.img = ti.Vector.field(3, float, self.res)
        self.norm = norm

    @ti.kernel
    def render(self, engine: ti.template(), image: ti.template()):
        for P in ti.grouped(image):
            if self.norm[P].norm_sqr() < eps:
                self.img[P] = 0
            else:
                self.render_at(engine, image, P)
        for P in ti.grouped(image):
            image[P] = lerp(0.5, self.img[P], image[P])

    @ti.func
    def render_at(self, engine, image: ti.template(), P):
        normal = self.norm[P]
        p = P + engine.bias[None]
        vpos = V23(engine.from_viewport(p), engine.depth[P] / engine.maxdepth)
        pos = mapply_pos(engine.V2W[None], vpos)
        viewdir = calc_viewdir(engine, p)

        material = tina.Mirror()
        odir, wei = material.sample(-viewdir, normal, 1)

        nsteps = 512

        step = 10 * (
                mapply_pos(engine.W2V[None], pos + odir / nsteps)
                - mapply_pos(engine.W2V[None], pos)).xy.norm()

        vtol = 32 * (mapply_pos(engine.W2V[None], pos - viewdir / nsteps
                ).z - mapply_pos(engine.W2V[None], pos).z)

        hit = 0
        D = float(P)
        pos += normal * 6e-3 - odir * noise(P) * step
        for i in range(nsteps):
            pos -= odir * step
            vpos = mapply_pos(engine.W2V[None], pos)
            if not all(-1 <= vpos <= 1):
                break
            D = engine.to_viewport(vpos)
            depth = engine.depth[int(D)] / engine.maxdepth
            if vpos.z - vtol < depth < vpos.z:
                hit = 1
                break

        res = bilerp(image, D) if hit else 0
        self.img[P] = res
