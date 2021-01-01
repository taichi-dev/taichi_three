from ..common import *


@ti.data_oriented
class WireframeRaster:
    def __init__(self, engine, maxwires=65536*3, linewidth=1.5, linecolor=(.9, .6, 0),
                 clipping=True, **extra_options):
        self.engine = engine
        self.res = self.engine.res
        self.maxfaces = maxwires
        self.linewidth = linewidth
        self.linecolor = tovector(linecolor)
        self.clipping = clipping

        self.nwires = ti.field(int, ())
        self.verts = ti.Vector.field(3, float, (maxwires, 2))

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

    @ti.kernel
    def render(self, shader: ti.template()):
        for f in ti.smart(self.get_wires_range()):
            Al, Bl = self.get_wire_vertices(f)
            Av, Bv = [self.engine.to_viewspace(p) for p in [Al, Bl]]

            if ti.static(self.clipping):
                if not all(-1 <= Av <= 1):
                    if not all(-1 <= Bv <= 1):
                        continue

            a, b = [self.engine.to_viewport(p) for p in [Av, Bv]]

            bot, top = ifloor(min(a, b) - self.linewidth), iceil(max(a, b) + self.linewidth)
            bot, top = max(bot, 0), min(top, self.res - 1)
            ban = (b - a).normalized()
            bann = (b - a).norm()
            wscale = 1 / ti.Vector([mapply(self.engine.W2V[None], p, 1)[1] for p in [Al, Bl]])
            for P in ti.grouped(ti.ndrange((bot.x, top.x + 1), (bot.y, top.y + 1))):
                pos = float(P) + self.engine.bias[None]
                cor = (pos - a).dot(ban)
                if -self.linewidth / 2 <= cor <= bann + self.linewidth / 2:
                    udf = abs((pos - a).cross(ban))
                    cor /= bann
                    if cor < 0:
                        udf = (pos - a).norm()
                        cor = 0
                    elif cor > 1:
                        udf = (pos - b).norm()
                        cor = 1
                    if udf < self.linewidth / 2:
                        wei = V(1 - cor, cor) * wscale
                        wei /= wei.x + wei.y
                        depth_f = wei.x * Av.z + wei.y * Bv.z
                        depth = int(depth_f * self.engine.maxdepth)
                        if ti.atomic_min(self.engine.depth[P], depth) > depth:
                            if self.engine.depth[P] >= depth:
                                shader.img[P] = self.linecolor