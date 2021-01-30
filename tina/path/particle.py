from ..advans import *
from .geometry import *


@ti.data_oriented
class ParticleTracer:
    @ti.func
    def calc_geometry(self, near, ind, uv, ro, rd):
        nrm = (ro - self.verts[ind]).normalized()
        return nrm, V(0., 0.)

    def __init__(self, maxpars=65536 * 16, coloring=False, multimtl=True, **extra_options):
        self.coloring = coloring
        self.multimtl = multimtl
        self.maxpars = maxpars

        self.verts = ti.Vector.field(3, float, maxpars)
        self.sizes = ti.field(float, maxpars)
        if self.coloring:
            self.colors = ti.Vector.field(3, float, maxpars)
        if self.multimtl:
            self.mtlids = ti.field(int, maxpars)
        self.npars = ti.field(int, ())

        @ti.materialize_callback
        def init_pars():
            self.sizes.fill(0.1)
            if self.coloring:
                self.colors.fill(1)

        self.tree = tina.BVHTree(self, self.maxpars * 4)

        self.eminds = ti.field(int, maxpars)
        self.neminds = ti.field(int, ())

    @ti.kernel
    def _export_geometry(self, verts: ti.ext_arr(), sizes: ti.ext_arr()):
        for i in range(self.npars[None]):
            sizes[i] = self.sizes[i]
            for k in ti.static(range(3)):
                verts[i, k] = self.verts[i][k]

    @ti.kernel
    def update_emission(self, mtltab: ti.template()):
        self.neminds[None] = 0
        for i in range(self.npars[None]):
            mtlid = self.get_material_id(i)
            material = mtltab.get(mtlid)
            emission = material.estimate_emission()
            if Vany(emission > 0):
                j = ti.atomic_add(self.neminds[None], 1)
                self.eminds[j] = i

    def update(self):
        pos = np.empty((self.npars[None], 3), dtype=np.float32)
        rad = np.empty((self.npars[None]), dtype=np.float32)
        self._export_geometry(pos, rad)
        rad = np.stack([rad, rad, rad], axis=1)
        self.tree.build(pos - rad, pos + rad)

    def clear_objects(self):
        self.npars[None] = 0

    @ti.kernel
    def add_object(self, pars: ti.template(), mtlid: ti.template()):
        pars.pre_compute()
        npars = pars.get_npars()
        base = self.npars[None]
        self.npars[None] += npars
        for i in range(self.npars[None]):
            j = base + i
            if ti.static(self.multimtl):
                self.mtlids[j] = mtlid
            vert = pars.get_particle_position(i)
            self.verts[j] = vert
            size = pars.get_particle_radius(i)
            self.sizes[j] = size
            if ti.static(self.coloring):
                color = pars.get_particle_color(i)
                self.colors[j] = color

    @ti.func
    def element_hit(self, ind, ro, rd):
        pos = self.verts[ind]
        rad = self.sizes[ind]
        hit, depth = ray_sphere_hit(pos, rad, ro, rd)
        return hit, depth, V(0., 0.)

    @ti.func
    def hit(self, ro, rd):
        return self.tree.hit(ro, rd)

    @ti.func
    def get_material_id(self, ind):
        if ti.static(not self.multimtl):
            return 0
        return self.mtlids[ind]

    @ti.func
    def sample_light_pos(self, org):
        pos, ind, wei = V3(0.), -1, 0.
        if self.neminds[None] != 0:
            ind = self.eminds[ti.random(int) % self.neminds[None]]
            orgdir = (org - self.verts[ind]).normalized()
            nrm = tangentspace(orgdir) @ spherical(ti.random(), ti.random())
            pos = nrm * self.sizes[ind] + self.verts[ind]
            wei = 2 * ti.pi * self.sizes[ind]**2
            wei *= max(0, nrm.dot((org - pos).normalized()))
            pos += nrm * eps * 8
        return pos, ind, wei * self.neminds[None]
