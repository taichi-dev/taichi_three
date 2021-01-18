from ..common import *


@ti.data_oriented
class WireframeRaster:
    def __init__(self, engine, maxwires=MAX, linewidth=1.5,
                  linecolor=(.9, .6, 0), clipping=False, **extra_options):
        self.engine = engine
        self.res = self.engine.res
        self.maxfaces = maxwires
        self.linewidth = linewidth
        self.clipping = clipping

        self.nwires = ti.field(int, ())
        self.verts = ti.Vector.field(3, float, (maxwires, 2))
        self.linecolor = ti.Vector.field(3, float, ())

        @ti.materialize_callback
        def init_linecolor():
            self.linecolor[None] = linecolor

    @ti.func
    def get_wires_range(self):
        for i in range(self.nwires[None]):
            yield i

    @ti.func
    def get_wire_vertices(self, f):
        A, B = self.verts[f, 0], self.verts[f, 1]
        return A, B

    @ti.kernel
    def set_object(self, mesh: ti.template()):
        mesh.pre_compute()
        self.nwires[None] = mesh.get_nfaces()
        for i in range(self.nwires[None]):
            verts = mesh.get_face_verts(i)
            for k in ti.static(range(2)):
                self.verts[i, k] = verts[k]

    @ti.kernel
    def set_wire_verts(self, verts: ti.ext_arr()):
        self.nfaces[None] = min(verts.shape[0], self.verts.shape[0])
        for i in range(self.nfaces[None]):
            for k in ti.static(range(2)):
                for l in ti.static(range(3)):
                    self.verts[i, k][l] = verts[i, k, l]

    @ti.func
    def draw_line(self, src, dst):
        dlt = dst - src
        adlt = abs(dlt)
        k, siz = V(1.0, 1.0), 0
        if adlt.x >= adlt.y:
            k.x = 1.0 if dlt.x >= 0 else -1.0
            k.y = k.x * dlt.y / dlt.x
            siz = int(adlt.x)
        else:
            k.y = 1.0 if dlt.y >= 0 else -1.0
            k.x = k.y * dlt.x / dlt.y
            siz = int(adlt.y)
        for i in range(siz + 1):
            pos = src + k * i
            yield pos, i / siz

    def render_occup(self):
        pass

    @ti.kernel
    def render_color(self, shader: ti.template()):
        for f in ti.smart(self.get_wires_range()):
            Al, Bl = self.get_wire_vertices(f)
            Av, Bv = [self.engine.to_viewspace(p) for p in [Al, Bl]]

            if ti.static(self.clipping):
                if not all(-1 <= Av <= 1):
                    if not all(-1 <= Bv <= 1):
                        continue

            a, b = [self.engine.to_viewport(p) for p in [Av, Bv]]

            ban = (b - a).normalized()
            wscale = 1 / ti.Vector([mapply(self.engine.W2V[None], p, 1)[1] for p in [Al, Bl]])
            for p, cor in ti.smart(self.draw_line(a, b)):
                pos = p + self.engine.bias[None]
                P = ifloor(pos)
                if all(0 <= P < self.res):
                    wei = V(1 - cor, cor) * wscale
                    wei /= wei.x + wei.y
                    depth_f = wei.x * Av.z + wei.y * Bv.z
                    depth = int(depth_f * self.engine.maxdepth)
                    if ti.atomic_min(self.engine.depth[P], depth) > depth:
                        if self.engine.depth[P] >= depth:
                            color = self.linecolor[None]
                            shader.blend_color(self.engine, P, pos, 1, color)
