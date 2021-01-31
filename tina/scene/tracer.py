from ..advans import *
from .raster import Scene


class MixedGeometryTracer:
    def __init__(self):
        self.tracers = []

    @ti.func
    def sample_light(self):
        if ti.static(len(self.tracers) == 1):
            pos, uv, ind, wei = self.tracers[0].sample_light()
            return pos, uv, ind, 0, wei

        gid = ti.random(int) % len(self.tracers)
        pos, ind, uv, wei = V(0., 0., 0.), V(0., 0., 0.), -1, V(0., 0., 0.)
        for i, tracer in ti.static(enumerate(self.tracers)):
            if i == gid:
                pos, nrm, ind, wei = tracer.sample_light()
        return pos, uv, ind, gid, wei

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
    def calc_geometry(self, gid, ind, uv, pos):
        if ti.static(len(self.tracers) == 1):
            return self.tracers[0].calc_geometry(ind, uv, pos)

        nrm, tex, mtlid = V(0., 0., 0.), V(0., 0.), -1
        for i, tracer in ti.static(enumerate(self.tracers)):
            if i == gid:
                nrm, tex, mtlid = tracer.calc_geometry(ind, uv, pos)
        return nrm, tex, mtlid


# noinspection PyMissingConstructor
class PTScene(Scene):
    def __init__(self, res=512, **options):
        self.mtltab = tina.MaterialTable()
        self.geom = MixedGeometryTracer()
        self.engine = tina.PathEngine(self.geom, self.mtltab, res)
        self.res = self.engine.res
        self.options = options

        self.materials = [tina.Lambert()]

        self.geom.tracers.append(tina.TriangleTracer(**self.options))

        @ti.materialize_callback
        def init_mtltab():
            self.mtltab.clear_materials()
            for material in self.materials:
                self.mtltab.add_material(material)

    def clear_objects(self):
        for tracer in self.geom.tracers:
            tracer.clear_objects()

    def add_mesh_object(self, world, verts, norms, coors, mtlid):
        self.geom.tracers[0].add_mesh(world, verts, norms, coors, mtlid)

    def clear(self):
        self.engine.clear_image()

    def update(self):
        self.engine.clear_image()
        for tracer in self.geom.tracers:
            tracer.update()
        for tracer in self.geom.tracers:
            tracer.update_emission(self.mtltab)

    def render(self, nsteps=10, russian=2):
        self.engine.trace(nsteps, russian)

    def render_light(self, nsteps=10, russian=2):
        self.engine.trace_light(nsteps, russian)

    @property
    def img(self):
        return self.engine.get_image()

    @property
    def raw_img(self):
        return self.engine.get_image(raw=True)

    def _fast_export_image(self, out):
        self.engine._fast_export_image(out)
