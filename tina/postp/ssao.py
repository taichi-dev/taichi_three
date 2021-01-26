from ..advans import *
from ..core.shader import calc_viewdir


@ti.data_oriented
class SSAO:
    def __init__(self, res, norm, nsamples=64, thresh=0.0,
            radius=0.2, factor=1.0, noise_size=4, taa=False):
        self.res = tovector(res)
        self.img = ti.field(float, self.res)
        self.radius = ti.field(float, ())
        self.thresh = ti.field(float, ())
        self.factor = ti.field(float, ())
        self.nsamples = ti.field(int, ())
        self.taa = taa
        self.norm = norm

        @ti.materialize_callback
        def init_params():
            self.radius[None] = radius
            self.thresh[None] = thresh
            self.factor[None] = factor
            self.nsamples[None] = nsamples if not self.taa else nsamples // 4

        if not self.taa:
            self.samples = ti.Vector.field(3, float, nsamples)
            self.rotations = ti.Vector.field(2, float, (noise_size, noise_size))
            ti.materialize_callback(self.seed_samples)

    @ti.kernel
    def seed_samples(self):
        for i in self.samples:
            self.samples[i] = self.make_sample()
        for I in ti.grouped(self.rotations):
            t = ti.tau * ti.random()
            self.rotations[I] = V(ti.cos(t), ti.sin(t))

    @ti.kernel
    def apply(self, out: ti.template()):
        for i, j in self.img:
            if ti.static(self.taa):
                out[i, j] *= 1 - self.img[i, j]
            else:
                r = 0.0
                rad = self.rotations.shape[0]
                offs = rad // 2
                for k, l in ti.ndrange(rad, rad):
                    r += self.img[i + k - offs, j + l - offs]
                out[i, j] *= 1 - r / rad**2

    @ti.func
    def make_sample(self):
        u, v = ti.random(), ti.random()
        r = lerp(ti.random()**1.5, 0.01, 1.0)
        u = lerp(u, 0.01, 1.0)
        return spherical(u, v) * r

    @ti.kernel
    def render(self, engine: ti.template()):
        for P in ti.grouped(engine.depth):
            self.render_at(engine, P)

    @ti.func
    def render_at(self, engine, P):
        normal = self.norm[P]
        p = P + engine.bias[None]
        vpos = V23(engine.from_viewport(p), engine.depth[P] / engine.maxdepth)
        pos = mapply_pos(engine.V2W[None], vpos)
        viewdir = calc_viewdir(engine, p)

        occ = 0.0
        radius = self.radius[None]
        vradius = engine.to_viewspace(pos - radius * viewdir).z - vpos.z
        for i in range(self.nsamples[None]):
            samp = V(0., 0., 0.)
            if ti.static(self.taa):
                samp = self.make_sample()
            else:
                samp = self.samples[i]
                rot = self.rotations[P % self.rotations.shape[0]]
                rotmat = ti.Matrix([[rot.x, rot.y], [-rot.x, rot.y]])
                samp.x, samp.y = rotmat @ samp.xy
            sample = tangentspace(normal) @ samp
            sample = pos + sample * radius
            sample = engine.to_viewspace(sample)
            D = engine.to_viewport(sample)
            if all(0 <= D < engine.res):
                depth = engine.depth[int(D)] / engine.maxdepth
                if depth < sample.z:
                    rc = vradius / (vpos.z - depth)
                    rc = smoothstep(abs(rc), 0, 1)
                    occ += rc

        ao = occ / self.nsamples[None]
        ao = self.factor[None] * (ao - self.thresh[None])
        self.img[P] = clamp(ao, 0, 1)


