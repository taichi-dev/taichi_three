import taichi as ti
import taichi_glsl as ts


@ti.data_oriented
class Geometry(ts.TaichiClass):
    @ti.func
    def render(self):
        for I in ti.grouped(ti.ndrange(*self.loop_range().shape())):
            self.subscript(I).do_render()

    def subscript(self, I):
        ret = self._subscript(I)
        try:
            ret.model = self.model
        except AttributeError:
            pass
        return ret

    def do_render(self):
        raise NotImplementedError


@ti.data_oriented
class Vertex(Geometry):
    @property
    def pos(self):
        return self.entries[0]

    @classmethod
    def _var(cls, shape=None):
        return ti.Vector.var(3, ti.f32, shape)

@ti.data_oriented
class VertexTex(Geometry):
    @property
    def pos(self):
        return self.entries[0]

    @classmethod
    def _var(cls, shape=None):
        return ti.Vector.var(2, ti.f32, shape)


@ti.data_oriented
class Line(Geometry):
    @property
    def idx(self):
        return self.entries[0]

    @classmethod
    def _var(cls, shape=None):
        return ti.Vector.var(2, ti.i32, shape)

    @ti.func
    def vertex(self, i: ti.template()):
        model = self.model
        return model.vertices[self.idx[i]]

    @ti.func
    def do_render(self):
        scene = self.model.scene
        W = 1
        A = scene.uncook_coor(scene.camera.untrans_pos(self.vertex(0).pos))
        B = scene.uncook_coor(scene.camera.untrans_pos(self.vertex(1).pos))
        M, N = int(ti.floor(min(A, B) - W)), int(ti.ceil(max(A, B) + W))
        for X in ti.grouped(ti.ndrange((M.x, N.x), (M.y, N.y))):
            P = B - A
            udf = (ts.cross(X, P) + ts.cross(B, A))**2 / P.norm_sqr()
            XoP = ts.dot(X, P)
            AoB = ts.dot(A, B)
            if XoP > B.norm_sqr() - AoB:
                udf = (B - X).norm_sqr()
            elif XoP < AoB - A.norm_sqr():
                    udf = (A - X).norm_sqr()
            if udf < 0:
                scene.img[X] = ts.vec3(1.0)
            elif udf < W**2:
                t = ts.smoothstep(udf, 0, W**2)
                ti.atomic_min(scene.img[X], ts.vec3(t))


@ti.data_oriented
class Face(Geometry):
    @property
    def idx(self):
        return self.entries[0]

    @classmethod
    def _var(cls, shape=None):
        return ti.Vector.var(6, ti.i32, shape)

    @ti.func
    def vertex(self, i: ti.template()):
        model = self.model
        return model.vertices[self.idx[i*2]]

    @ti.func
    def texture_pos(self, i: ti.template()):
        model = self.model
        return model.vertex_tex[self.idx[i*2+1]]

    @ti.func
    def texture(self, x: ti.template(), y: ti.template()):
        model = self.model
        X = ti.max(0, ti.min(1, x))
        Y = ti.max(0, ti.min(1, y))
        X = X * model.texture.shape[1]
        Y = Y * model.texture.shape[0]
        return model.texture[Y, X, :]

    @ti.func
    def do_render(self):
        model = self.model
        scene = model.scene
        L2W = model.L2W
        a = scene.camera.untrans_pos(L2W @ self.vertex(0).pos)
        b = scene.camera.untrans_pos(L2W @ self.vertex(1).pos)
        c = scene.camera.untrans_pos(L2W @ self.vertex(2).pos)

        t_a = self.texture_pos(0).pos
        t_b = self.texture_pos(1).pos
        t_c = self.texture_pos(2).pos

        A = scene.uncook_coor(a)
        B = scene.uncook_coor(b)
        C = scene.uncook_coor(c)
        B_A = B - A
        C_B = C - B
        A_C = A - C
        ilB_A = 1 / ts.length(B_A)
        ilC_B = 1 / ts.length(C_B)
        ilA_C = 1 / ts.length(A_C)
        B_A *= ilB_A
        C_B *= ilC_B
        A_C *= ilA_C
        BxA = ts.cross(B, A) * ilB_A
        CxB = ts.cross(C, B) * ilC_B
        AxC = ts.cross(A, C) * ilA_C
        normal = ts.normalize(ts.cross(a - c, a - b))
        light_dir = scene.camera.untrans_dir(scene.light_dir[None])
        pos = (a + b + c) / 3
        color = scene.opt.render_func(pos, normal, ts.vec3(0.0), light_dir)
        # color = scene.opt.pre_process(color)

        Ak = 1 / (ts.cross(A, C_B) + CxB)
        Bk = 1 / (ts.cross(B, A_C) + AxC)
        Ck = 1 / (ts.cross(C, B_A) + BxA)

        W = 1
        ZW = ts.distance(a, b) * 0.2
        M, N = int(ti.floor(min(A, B, C) - W)), int(ti.ceil(max(A, B, C) + W))
        for X in ti.grouped(ti.ndrange((M.x, N.x), (M.y, N.y))):
            if X.x < 0 or X.x >= scene.res[0] or X.y < 0 or X.y >= scene.res[1]:
                continue
            AB = ts.cross(X, B_A) + BxA
            BC = ts.cross(X, C_B) + CxB
            CA = ts.cross(X, A_C) + AxC
            udf = max(AB, BC, CA)
            if udf < W:
                w_A = max(1e-6, Ak*BC)
                w_B = max(1e-6, Bk*CA)
                w_C = max(1e-6, Ck*AB)
                w_sum = w_A + w_B + w_C
                zindex = (w_A * a.z + w_B * b.z + w_C * c.z) / w_sum
                xindex = (w_A * t_a.x + w_B * t_b.x + w_C * t_c.x) / w_sum
                yindex = (w_A * t_a.y + w_B * t_b.y + w_C * t_c.y) / w_sum
                xindex = xindex * self.model.texture.shape[1]
                yindex = yindex * self.model.texture.shape[0]
                tex_color = self.model.texture[self.model.texture.shape[0] - int(yindex), int(xindex)]
                new_color = tex_color * (ts.vec3(scene.opt.diffuse) + color)
                zstep = zindex - ti.atomic_max(scene.zbuf[X], zindex)
                if zstep >= 0:
                    scene.img[X] = new_color
