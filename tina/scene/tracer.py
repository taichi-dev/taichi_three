from ..advans import *
from .raster import Scene


class MixedGeometryTracer:
    def __init__(self):
        self.tracers = []

    @ti.func
    def hit(self, ro, rd):
        if ti.static(len(self.tracers) == 1):
            near, ind, uv = self.tracers[0].hit(ro, rd)
            gid = -1 if ind == -1 else 0
            return near, ind, gid, uv

        ret_near, ret_ind, ret_gid, ret_uv = inf, -1, -1, V(0., 0.)
        for gid, tracer in ti.static(enumerate(self.tracers)):
            near, ind, uv = tracer.hit(ro, rd)
            if near < ret_near:
                    ret_near, ret_ind, ret_gid, ret_uv = near, ind, gid, uv
        return ret_near, ret_ind, ret_gid, ret_uv

    @ti.func
    def calc_geometry(self, near, gid, ind, uv, ro, rd):
        if ti.static(len(self.tracers) == 1):
            return self.tracers[0].calc_geometry(near, ind, uv, ro, rd)

        nrm, tex = V(0., 0., 0.), V(0., 0.)
        for i, tracer in ti.static(enumerate(self.tracers)):
            if i == gid:
                nrm, tex = tracer.calc_geometry(near, ind, uv, ro, rd)
        return nrm, tex

    @ti.func
    def get_material_id(self, ind, gid):
        if ti.static(len(self.tracers) == 1):
            return self.tracers[0].get_material_id(ind)

        mtlid = -1
        for i, tracer in ti.static(enumerate(self.tracers)):
            if i == gid:
                mtlid = tracer.get_material_id(ind)
        return mtlid


# noinspection PyMissingConstructor
class PTScene(Scene):
    def __init__(self, res=512, **options):
        self.mtltab = tina.MaterialTable()
        self.lighting = tina.PathLighting()
        self.geom = MixedGeometryTracer()
        self.engine = tina.PathEngine(self.geom, self.lighting, self.mtltab, res)
        self.res = self.engine.res
        self.options = options

        self.default_material = tina.Lambert()
        self.materials = []
        self.objects = []
        self.tracers = []

        @ti.materialize_callback
        def init_mtltab():
            self.mtltab.clear_materials()
            for material in self.materials:
                self.mtltab.add_material(material)

    def add_object(self, object, material=None, tracer=None):
        if material is None:
            material = self.default_material
        if tracer is None:
            if hasattr(object, 'get_nfaces'):
                if not hasattr(self, 'triangle_tracer'):
                    self.triangle_tracer = tina.TriangleTracer(**self.options)
                tracer = self.triangle_tracer
            elif hasattr(object, 'get_npars'):
                if not hasattr(self, 'particle_tracer'):
                    self.particle_tracer = tina.ParticleTracer(**self.options)
                tracer = self.particle_tracer
            else:
                raise ValueError(f'cannot determine tracer type of object: {object}')

        if material not in self.materials:
            self.materials.append(material)
        if tracer not in self.geom.tracers:
            self.geom.tracers.append(tracer)
        mtlid = self.materials.index(material)
        self.objects.append((object, mtlid, tracer))

    def clear(self):
        self.engine.clear_image()

    def update(self):
        self.engine.clear_image()
        for tracer in self.geom.tracers:
            tracer.clear_objects()
        for object, mtlid, tracer in self.objects:
            tracer.add_object(object, mtlid)
        for tracer in self.geom.tracers:
            tracer.update()

    def render(self, nsteps=5, strict=True):
        self.engine.load_rays()
        for step in range(nsteps):
            self.engine.step_rays()
        self.engine.update_image(strict)

    @property
    def img(self):  # TODO: use postp for this too
        return self.engine.get_image()

    @property
    def raw_image(self):
        return self.engine.get_image(lambda x: x)
