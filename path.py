from tina.advans import *

ti.init(ti.cpu)

rough = ti.field(float, ())


@ti.func
def dot_or_zero(a, b, zero=0):
    return max(zero, a.dot(b))


@ti.func
def mis_power_heuristic(pf, pg):
    f = pf**2
    g = pg**2
    return f / (f + g)


@ti.func
def pbr_fresnel(metallic, albedo, specular=0.5):
    f0 = metallic * albedo + (1 - metallic) * 0.16 * specular**2
    return f0


@ti.func
def pbr_brdf(f0, roughness, nrm, idir, odir):
    half = (idir + odir).normalized()
    NoH = dot_or_zero(half, nrm, 1e-10)
    NoL = dot_or_zero(idir, nrm, 1e-10)
    NoV = dot_or_zero(odir, nrm, 1e-10)
    VoH = dot_or_zero(half, odir, 1e-10)
    LoH = dot_or_zero(half, idir, 1e-10)

    alpha2 = max(0, roughness**2)
    denom = 1 - NoH**2 * (1 - alpha2)
    ndf = alpha2 / denom**2

    k = (roughness + 1)**2 / 8
    vdf = 0.5 / ((NoV * k + 1 - k))
    vdf *= 0.5 / ((NoL * k + 1 - k))

    fdf = f0 + (1 - f0) * (1 - VoH)**5

    return fdf * vdf * ndf

@ti.func
def pbr_sample(f0, roughness, nrm, idir, u, v):
    alpha2 = max(0, roughness**2)

    u = ti.sqrt((1 - u) / (1 - u * (1 - alpha2)))
    rdir = reflect(-idir, nrm)
    axes = tangentspace(rdir)
    odir = axes @ spherical(u, v)

    half = (idir + odir).normalized()
    VoH = dot_or_zero(half, odir, 1e-10)
    fdf = f0 + (1 - f0) * (1 - VoH)**5
    if odir.dot(nrm) < 0:
        odir = -odir
        fdf = 0.0

    return odir, fdf


@ti.data_oriented
class BVHTree:
    def __init__(self, geom, N_tree=MAX, dim=3):
        self.geom = geom
        self.N_tree = N_tree
        self.dim = dim

        self.dir = ti.field(int)
        self.min = ti.Vector.field(self.dim, float)
        self.max = ti.Vector.field(self.dim, float)
        self.ind = ti.field(int)
        self.tree = ti.root.dense(ti.i, self.N_tree)
        self.tree.place(self.dir, self.min, self.max, self.ind)

    def build(self, pmin, pmax):
        assert len(pmin) == len(pmax)
        assert np.all(pmax >= pmin)
        data = lambda: None
        data.dir = self.dir.to_numpy()
        data.dir[:] = -1
        data.min = self.min.to_numpy()
        data.max = self.max.to_numpy()
        data.ind = self.ind.to_numpy()
        print('[Tina] building tree...')
        self._build(data, pmin, pmax, np.arange(len(pmin)), 1)
        self._build_from_data(data.dir, data.min, data.max, data.ind)
        print('[Tina] building tree done')

    @ti.kernel
    def _build_from_data(self,
            data_dir: ti.ext_arr(),
            data_min: ti.ext_arr(),
            data_max: ti.ext_arr(),
            data_ind: ti.ext_arr()):
        for i in range(self.dir.shape[0]):
            if data_dir[i] == -1:
                continue
            self.dir[i] = data_dir[i]
            for k in ti.static(range(self.dim)):
                self.min[i][k] = data_min[i, k]
                self.max[i][k] = data_max[i, k]
            self.ind[i] = data_ind[i]

    def _build(self, data, pmin, pmax, pind, curr):
        assert curr < self.N_tree, curr
        if not len(pind):
            return

        elif len(pind) <= 1:
            data.dir[curr] = 0
            data.ind[curr] = pind[0]
            data.min[curr] = pmin[0]
            data.max[curr] = pmax[0]
            return

        bmax = np.max(pmax, axis=0)
        bmin = np.min(pmin, axis=0)
        dir = np.argmax(bmax - bmin)
        sort = np.argsort(pmax[:, dir] + pmin[:, dir])
        mid = len(sort) // 2
        lsort = sort[:mid]
        rsort = sort[mid:]

        lmin, rmin = pmin[lsort], pmin[rsort]
        lmax, rmax = pmax[lsort], pmax[rsort]
        lind, rind = pind[lsort], pind[rsort]
        data.dir[curr] = 1 + dir
        data.ind[curr] = 0
        data.min[curr] = bmin
        data.max[curr] = bmax
        self._build(data, lmin, lmax, lind, curr * 2)
        self._build(data, rmin, rmax, rind, curr * 2 + 1)

    @ti.kernel
    def _active_indices(self, out: ti.ext_arr()):
        for curr in self.dir:
            if self.dir[curr] != 0:
                out[curr] = 1

    def active_indices(self):
        ind = np.zeros(self.N_tree, dtype=np.int32)
        self._active_indices(ind)
        return np.bool_(ind)

    def visualize(self, gui):
        assert self.dim == 2
        bmin = self.min.to_numpy()
        bmax = self.max.to_numpy()
        ind = self.active_indices()
        bmin, bmax = bmin[ind], bmax[ind]
        delta = bmax - bmin
        ind = np.any(delta >= 0.02, axis=1)
        bmin, bmax = bmin[ind], bmax[ind]
        bmin = bmin * 0.5 + 0.5
        bmax = bmax * 0.5 + 0.5
        gui.rects(bmin, bmax, color=0xff0000)

    @ti.func
    def hit(self, ro, rd):
        stack = tina.Stack.instance()
        near = inf
        ntimes = 0
        stack.clear()
        stack.push(1)
        hitind = -1
        hituv = V(0., 0.)
        while ntimes < self.N_tree and stack.size() != 0:
            curr = stack.pop()

            if self.dir[curr] == 0:
                ind = self.ind[curr]
                hit, depth, uv = self.geom.element_hit(ind, ro, rd)
                if hit != 0 and depth < near:
                    near = depth
                    hitind = ind
                    hituv = uv
                continue

            bmin, bmax = self.min[curr], self.max[curr]
            bnear, bfar = ray_aabb_hit(bmin, bmax, ro, rd)
            if bnear > bfar:
                continue

            ntimes += 1
            stack.push(curr * 2)
            stack.push(curr * 2 + 1)
        return near, hitind, hituv


@ti.data_oriented
class Triangles:
    def __init__(self, maxfaces=MAX, smoothing=False, texturing=False,
                 **extra_options):
        self.smoothing = smoothing
        self.texturing = texturing
        self.maxfaces = maxfaces

        self.verts = ti.Vector.field(3, float, (maxfaces, 3))
        if self.smoothing:
            self.norms = ti.Vector.field(3, float, (maxfaces, 3))
        if self.texturing:
            self.coors = ti.Vector.field(2, float, (maxfaces, 3))
        self.mtlids = ti.field(int, maxfaces)
        self.nfaces = ti.field(int, ())

        self.tree = tina.BVHTree(self, self.maxfaces * 4)

    def clear_objects(self):
        self.nfaces[None] = 0

    def add_object(self, mesh, mtlid):
        @ti.materialize_callback
        def _():
            obj = tina.export_simple_mesh(mesh)
            self.add_mesh(np.eye(4, dtype=np.float32), obj['fv'], obj['fn'], obj['ft'], mtlid)

    @ti.kernel
    def add_mesh(self, world: ti.ext_arr(), verts: ti.ext_arr(),
            norms: ti.ext_arr(), coors: ti.ext_arr(), mtlid: int):
        trans = ti.Matrix.zero(float, 4, 4)
        trans_norm = ti.Matrix.zero(float, 3, 3)
        for i, j in ti.static(ti.ndrange(4, 4)):
            trans[i, j] = world[i, j]
        for i, j in ti.static(ti.ndrange(3, 3)):
            trans_norm[i, j] = world[i, j]
        trans_norm = trans_norm.inverse().transpose()
        nfaces = verts.shape[0]
        base = self.nfaces[None]
        self.nfaces[None] += nfaces
        for i in range(nfaces):
            j = base + i
            self.mtlids[j] = mtlid
            for k in ti.static(range(3)):
                for l in ti.static(range(3)):
                    self.verts[j, k][l] = verts[i, k, l]
                self.verts[j, k] = mapply_pos(trans, self.verts[j, k])
            if ti.static(self.smoothing):
                for k in ti.static(range(3)):
                    for l in ti.static(range(3)):
                        self.norms[j, k][l] = norms[i, k, l]
                    self.norms[j, k] = trans_norm @ self.norms[j, k]
            if ti.static(self.texturing):
                for k in ti.static(range(3)):
                    for l in ti.static(range(2)):
                        self.coors[j, k][l] = coors[i, k, l]

    @ti.kernel
    def _export_vertices(self, verts: ti.ext_arr()):
        for i in range(self.nfaces[None]):
            for k in ti.static(range(3)):
                for l in ti.static(range(3)):
                    verts[i, k, l] = self.verts[i, k][l]

    def update(self):
        verts = np.empty((self.nfaces[None], 3, 3), dtype=np.float32)
        self._export_vertices(verts)
        bmax = np.max(verts, axis=1)
        bmin = np.min(verts, axis=1)
        self.tree.build(bmin, bmax)

    @ti.func
    def hit(self, ro, rd):
        return self.tree.hit(ro, rd)

    @ti.func
    def get_material_id(self, ind):
        return self.mtlids[ind]

    @ti.func
    def calc_geometry(self, ind, uv):
        nrm = V(0., 0., 0.)
        tex = V(0., 0.)
        wei = V(1 - uv.x - uv.y, uv.x, uv.y)
        mtlid = self.mtlids[ind]

        if ti.static(self.texturing):
            c0 = self.coors[ind, 0]
            c1 = self.coors[ind, 1]
            c2 = self.coors[ind, 2]
            tex = c0 * wei.x + c1 * wei.y + c2 * wei.z

        if ti.static(self.smoothing):
            n0 = self.norms[ind, 0]
            n1 = self.norms[ind, 1]
            n2 = self.norms[ind, 2]
            nrm = (n0 * wei.x + n1 * wei.y + n2 * wei.z).normalized()
        else:
            v0 = self.verts[ind, 0]
            v1 = self.verts[ind, 1]
            v2 = self.verts[ind, 2]
            nrm = (v1 - v0).cross(v2 - v0).normalized()

        return nrm, tex, mtlid

    @ti.func
    def element_hit(self, ind, ro, rd):
        v0 = self.verts[ind, 0]
        v1 = self.verts[ind, 1]
        v2 = self.verts[ind, 2]
        hit, depth, uv = tina.ray_triangle_hit(v0, v1, v2, ro, rd)
        return hit, depth, uv


@ti.data_oriented
class PathEngine:
    def __init__(self, geom, res=512):
        if isinstance(res, int):
            res = res, res

        self.res = tovector(res)
        self.nrays = self.res.x * self.res.y

        self.img = ti.Vector.field(3, float, self.res)
        self.cnt = ti.field(int, self.res)

        self.geom = geom
        self.stack = tina.Stack(N_mt=self.nrays)

        self.W2V = ti.Matrix.field(4, 4, float, ())
        self.V2W = ti.Matrix.field(4, 4, float, ())

        @ti.materialize_callback
        @ti.kernel
        def init_engine():
            self.W2V[None] = ti.Matrix.identity(float, 4)
            self.W2V[None][2, 2] = -1
            self.V2W[None] = ti.Matrix.identity(float, 4)
            self.V2W[None][2, 2] = -1

    def set_camera(self, view, proj):
        W2V = proj @ view
        V2W = np.linalg.inv(W2V)
        self.W2V.from_numpy(np.array(W2V, dtype=np.float32))
        self.V2W.from_numpy(np.array(V2W, dtype=np.float32))

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
    def generate_ray(self, I):
        bias = V(ti.random(), ti.random())
        uv = (I + bias) / self.res * 2 - 1
        ro = mapply_pos(self.V2W[None], V(uv.x, uv.y, -1.0))
        ro1 = mapply_pos(self.V2W[None], V(uv.x, uv.y, +1.0))
        rd = (ro1 - ro).normalized()
        return ro, rd

    @property
    def lambertian_brdf(self):
        return 1 / ti.pi

    @ti.func
    def visible_to_light(self, pos, dir):
        visible = 0
        light, light_dis = self.hit_light(pos, dir)
        if light:
            hit_dis, hit_ind, hit_uv = self.geom.hit(pos + dir * eps * 4, dir)
            if hit_ind == -1 or light_dis < hit_dis:
                visible = 1
        return visible

    @ti.func
    def hit_light(self, pos, dir):
        near, far = tina.ray_aabb_hit(V(-0.5, -0.5, 4.0), V(0.5, 0.5, 4.1), pos, dir)
        return 0 < near < far, near

    @ti.func
    def sample_area_light(self, pos, nrm):
        x = ti.random() - 0.5
        y = ti.random() - 0.5
        on_light_pos = V(x, y, 4.0)
        return (on_light_pos - pos).normalized()

    @ti.func
    def compute_area_light_pdf(self, pos, dir):
        hit, hit_dis = self.hit_light(pos, dir)
        pdf = 0.0
        if hit:
            light_area = 1
            light_nrm = V(0.0, 0.0, 1.0)
            l_cos = light_nrm.dot(dir)
            if l_cos > 0:
                pdf = (dir * hit_dis).norm_sqr() / (light_area * l_cos)
        return pdf

    @ti.func
    def compute_brdf_pdf(self, nrm, outdir, indir):
        brdf = pbr_brdf(1.0, rough[None], nrm, -indir, outdir)
        return brdf * dot_or_zero(nrm, outdir) / ti.pi

    @ti.func
    def sample_brdf(self, nrm, indir):
        u, v = ti.random(), ti.random()
        outdir, weight = pbr_sample(1.0, rough[None], nrm, -indir, u, v)
        return outdir

    @ti.func
    def sample_ray_dir(self, indir, hit_nrm, hit_pos):
        outdir = self.sample_brdf(hit_nrm, indir)
        pdf = dot_or_zero(hit_nrm, outdir) / ti.pi
        return outdir, pdf

    @ti.func
    def sample_direct_light(self, hit_pos, hit_nrm, indir):
        direct_li = V3(0.0)

        fl = self.lambertian_brdf

        to_light_dir = self.sample_area_light(hit_pos, hit_nrm)
        if to_light_dir.dot(hit_nrm) > 0:
            light_pdf = self.compute_area_light_pdf(hit_pos, to_light_dir)
            brdf_pdf = self.compute_brdf_pdf(hit_nrm, to_light_dir, indir)
            if light_pdf > 0 and brdf_pdf > 0:
                if self.visible_to_light(hit_pos, to_light_dir):
                    w = mis_power_heuristic(light_pdf, brdf_pdf)
                    ranprint(w, light_pdf, brdf_pdf)
                    nl = dot_or_zero(to_light_dir, hit_nrm)
                    direct_li += fl * w * nl / light_pdf * U3(0)

        return direct_li * 64

    @ti.kernel
    def render(self):
        for i in ti.smart(self.stack):
            I = V(i % self.res.x, i // self.res.x)
            pos, dir = self.generate_ray(I)

            acc_color = V3(0.0)
            throughput = V3(1.0)
            for depth in range(10):

                hit_light, _ = self.hit_light(pos, dir)
                if hit_light:
                    acc_color += throughput * 64 * U3(2)

                hit_dis, hit_ind, hit_uv = self.geom.hit(pos, dir)
                if hit_ind == -1:
                    #background = lerp(unlerp(dir.y, -1, 1), V(.2, .5, 1.), 1.)
                    #acc_color += throughput * background
                    break
                hit_nrm, hit_tex, hit_mtl = self.geom.calc_geometry(hit_ind, hit_uv)
                hit_pos = pos + hit_dis * dir + hit_nrm * eps * 8

                direct_li = self.sample_direct_light(hit_pos, hit_nrm, dir)
                acc_color += throughput * direct_li

                outdir, pdf = self.sample_ray_dir(dir, hit_nrm, hit_pos)

                brdf = self.lambertian_brdf
                throughput *= brdf * dot_or_zero(hit_nrm, outdir) / pdf

                pos = hit_pos
                dir = outdir

            self.img[I] += acc_color
            self.cnt[I] += 1


geom = Triangles(smoothing=True)
engine = PathEngine(geom)
obj = tina.readobj('assets/cube.obj')
geom.add_mesh(np.eye(4, dtype=np.float32), obj['v'][obj['f'][:, :, 0]], obj['vn'][obj['f'][:, :, 2]], obj['vt'][obj['f'][:, :, 1]], 0)
geom.update()

gui = ti.GUI()
rough.slider = gui.slider('rough', 0, 1, 0.1)
rough.slider.value = 0.1
ctl = tina.Control(gui)
while gui.running:
    rough[None] = rough.slider.value
    if ctl.process_events():
        engine.clear_image()
    ctl.apply_camera(engine)
    engine.render()
    gui.set_image(engine.get_image())
    gui.show()
