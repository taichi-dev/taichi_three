import taichi as ti
import numpy as np
import base64


@ti.data_oriented
class MCISO:
    def __init__(self, N=128, N_res=None, dim=3, use_sparse=False, has_normal=True):
        self.N = N
        self.dim = dim
        self.N_res = N_res or N**self.dim
        self.use_sparse = use_sparse
        self.has_normal = has_normal

        et = [_et2, _et3][dim - 2]
        self.et = ti.Vector.field(dim, int, et.shape[:2])
        @ti.materialize_callback
        def init_tables():
            self.et.from_numpy(et)

        self.m = ti.field(float)
        self.g = ti.Vector.field(self.dim, float)
        self.Jtab = ti.field(int)
        indices = [ti.ij, ti.ijk][dim - 2]
        ti.root.dense(indices, self.N).place(self.g)
        if self.use_sparse:
            ti.root.pointer(indices, self.N // 16).dense(indices, 16).place(self.m)
            ti.root.dense(indices, 1).dense(ti.l, self.dim).pointer(indices, self.N // 8).bitmasked(indices, 8).place(self.Jtab)
        else:
            ti.root.dense(indices, 1).dense(ti.l, self.dim).dense(indices, self.N).place(self.Jtab)
            ti.root.dense(indices, self.N).place(self.m)

        self.Js = ti.Vector.field(self.dim + 1, int, (self.N_res, self.dim))
        self.Jts = ti.Vector.field(self.dim, int, self.N_res)
        self.vs = ti.Vector.field(self.dim, float, self.N_res)
        if self.has_normal:
            self.ns = ti.Vector.field(self.dim, float, self.N_res)
        self.Js_n = ti.field(int, ())
        self.vs_n = ti.field(int, ())

    @ti.kernel
    def march(self):
        self.Js_n[None] = 0
        self.vs_n[None] = 0

        if ti.static(self.has_normal):
            for I in ti.grouped(self.g):
                r = ti.Vector.zero(float, self.dim)
                for i in ti.static(range(self.dim)):
                    d = ti.Vector.unit(self.dim, i, int)
                    r[i] = self.m[I + d] - self.m[I - d]
                self.g[I] = -r.normalized(1e-4)

        for I in ti.grouped(self.m):
            id = self.get_cubeid(I)
            for m in range(self.et.shape[1]):
                et = self.et[id, m]
                if et[0] == -1:
                    break

                Js_n = ti.atomic_add(self.Js_n[None], 1)
                for l in ti.static(range(self.dim)):
                    e = et[l]
                    J = ti.Vector(I.entries + [0])
                    if ti.static(self.dim == 2):
                        if e == 1 or e == 2: J.z = 1
                        if e == 2: J.x += 1
                        if e == 3: J.y += 1
                    else:
                        if e == 1 or e == 3 or e == 5 or e == 7: J.w = 1
                        elif e == 8 or e == 9 or e == 10 or e == 11: J.w = 2
                        if e == 1 or e == 5 or e == 9 or e == 10: J.x += 1
                        if e == 2 or e == 6 or e == 10 or e == 11: J.y += 1
                        if e == 4 or e == 5 or e == 6 or e == 7: J.z += 1
                    self.Js[Js_n, l] = J
                    self.Jtab[J] = 1

        for J in ti.grouped(self.Jtab):
            if not ti.static(self.use_sparse):
                if self.Jtab[J] == 0:
                    continue
            vs_n = ti.atomic_add(self.vs_n[None], 1)
            I = ti.Vector(ti.static(J.entries[:-1]))
            vs = I * 1.0
            ns = self.g[I]
            p1 = self.m[I]
            for t in ti.static(range(self.dim)):
                if J.entries[-1] == t:
                    K = I + ti.Vector.unit(self.dim, t, int)
                    p2 = self.m[K]
                    n2 = self.g[K]
                    p = (1 - p1) / (p2 - p1)
                    p = max(0, min(1, p))
                    vs[t] += p
                    ns += p * n2
            self.vs[vs_n] = vs
            if ti.static(self.has_normal):
                self.ns[vs_n] = ns.normalized(1e-4)
            self.Jtab[J] = vs_n

        for i in range(self.Js_n[None]):
            for l in ti.static(range(self.dim)):
                self.Jts[i][l] = self.Jtab[self.Js[i, l]]

    def clear(self):
        if self.use_sparse:
            ti.root.deactivate_all()
        else:
            self.m.fill(0)
            self.Jtab.fill(0)

    @ti.func
    def get_cubeid(self, I):
        id = 0
        if ti.static(self.dim == 2):
            i, j = I
            if self.m[i, j] > 1: id |= 1
            if self.m[i + 1, j] > 1: id |= 2
            if self.m[i, j + 1] > 1: id |= 4
            if self.m[i + 1, j + 1] > 1: id |= 8
        else:
            i, j, k = I
            if self.m[i, j, k] > 1: id |= 1
            if self.m[i + 1, j, k] > 1: id |= 2
            if self.m[i + 1, j + 1, k] > 1: id |= 4
            if self.m[i, j + 1, k] > 1: id |= 8
            if self.m[i, j, k + 1] > 1: id |= 16
            if self.m[i + 1, j, k + 1] > 1: id |= 32
            if self.m[i + 1, j + 1, k + 1] > 1: id |= 64
            if self.m[i, j + 1, k + 1] > 1: id |= 128
        return id

    def get_mesh(self):
        ret = {}
        ret['f'] = self.Jts.to_numpy()[:self.Js_n[None]][:, ::-1]
        ret['v'] = (self.vs.to_numpy()[:self.vs_n[None]] + 0.5) / self.N
        if self.has_normal:
            ret['vn'] = self.ns.to_numpy()[:self.vs_n[None]]
        return ret

    @ti.func
    def pre_compute(self):
        pass

    @ti.func
    def get_nfaces(self):
        return self.Js_n[None]

    @ti.func
    def get_face_verts(self, n):
        a = (self.vs[self.Jts[n][2]] + 0.5) / self.N
        b = (self.vs[self.Jts[n][1]] + 0.5) / self.N
        c = (self.vs[self.Jts[n][0]] + 0.5) / self.N
        return a, b, c

    @ti.func
    def get_face_norms(self, n):
        a = self.ns[self.Jts[n][2]]
        b = self.ns[self.Jts[n][1]]
        c = self.ns[self.Jts[n][0]]
        return a, b, c


def extend_bounds(m, bound=1):
    assert len(m.shape) == 3
    x = np.zeros((bound, m.shape[1], m.shape[2]), dtype=m.dtype)
    y = np.zeros((m.shape[0] + bound * 2, bound, m.shape[2]), dtype=m.dtype)
    z = np.zeros((m.shape[0] + bound * 2, m.shape[1] + bound * 2, bound), dtype=m.dtype)
    m = np.concatenate([x, m, x], axis=0)
    m = np.concatenate([y, m, y], axis=1)
    m = np.concatenate([z, m, z], axis=2)
    return m


class MCISO_Example(MCISO):
    @ti.func
    def gauss(self, x):
        return ti.exp(-6 * x**2)

    @ti.kernel
    def touch(self, mx: float, my: float):
        for o in ti.grouped(ti.ndrange(*[self.N] * self.dim)):
            p = o / self.N
            a = self.gauss((p - 0.5).norm() / 0.25)
            p.x -= mx - 0.5 / self.N
            p.y -= my - 0.5 / self.N
            if ti.static(self.dim == 3):
                p.z -= 0.5
            b = self.gauss(p.norm() / 0.25)
            r = max(a + b - 0.08, 0)
            if r <= 0:
                continue
            self.m[o] = r * 3

    def main(self):
        gui = ti.GUI('Marching cube')
        while gui.running and not gui.get_event(gui.ESCAPE):
            self.clear()
            self.touch(*gui.get_cursor_pos())
            self.march()
            Jts = self.Jts.to_numpy()[:self.Js_n[None]]
            vs = (self.vs.to_numpy()[:self.vs_n[None]] + 0.5) / self.N
            ret = vs[Jts]
            if self.dim == 2:
                #gui.set_image(ti.imresize(self.m, *gui.res))
                gui.set_image(ti.imresize(self.g, *gui.res) + 0.5)
                gui.lines(ret[:, 0], ret[:, 1], color=0xff66cc, radius=1.25)
                #gui.circles(vs, color=0xffff33, radius=1.5)
            else:
                #gui.triangles(ret[:, 0, 0:2],
                              #ret[:, 1, 0:2],
                              #ret[:, 2, 0:2],
                              #color=0xffcc66)
                gui.lines(ret[:, 0, 0:2],
                          ret[:, 1, 0:2],
                          color=0xff66cc,
                          radius=0.5)
                gui.lines(ret[:, 1, 0:2],
                          ret[:, 2, 0:2],
                          color=0xff66cc,
                          radius=0.5)
                gui.lines(ret[:, 2, 0:2],
                          ret[:, 0, 0:2],
                          color=0xff66cc,
                          radius=0.5)
                gui.text(f'Press space to save mesh to PLY ({len(vs)} verts, {len(Jts)} faces)', (0, 1))
                if gui.is_pressed(gui.SPACE):
                    writer = ti.PLYWriter(num_vertices=len(vs), num_faces=len(Jts))
                    writer.add_vertex_pos(vs[:, 0], vs[:, 1], vs[:, 2])
                    writer.add_faces(Jts)
                    writer.export('mciso_output.ply')
                    print('Mesh saved to mciso_output.ply')
            gui.show()


@ti.data_oriented
class Voxelizer:
    def __init__(self, N, pmin=0, pmax=1, radius=1, weight=50):
        self.N = N
        self.pmin = ti.Vector([pmin for i in range(3)])
        self.pmax = ti.Vector([pmax for i in range(3)])

        self.weight = weight
        self.radius = radius
        if self.radius:
            self.tmp1 = ti.field(float, (N, N, N))
            self.tmp2 = ti.field(float, (N, N, N))
            self.gwei = ti.field(float, self.radius + 1)

            @ti.materialize_callback
            @ti.kernel
            def init_gwei():
                sum = -1.0
                for i in self.gwei:
                    x = i / self.radius
                    y = ti.exp(-x**2)
                    self.gwei[i] = y
                    sum += y * 2
                for i in self.gwei:
                    self.gwei[i] /= sum

    @ti.kernel
    def voxelize(self, out: ti.template(), pos: ti.template()):
        for i in pos:
            p = (pos[i] - self.pmin) / (self.pmax - self.pmin)
            Xp = p * self.N
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            for offset in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                dpos = (offset - fx) / self.N
                weight = float(self.weight)
                for t in ti.static(range(3)):
                    weight *= w[offset[t]][t]
                out[base + offset] += weight
        if ti.static(self.radius):
            tmp = ti.static([out, self.tmp1, self.tmp2, out])
            for di in ti.static(range(3)):
                dir = ti.static(ti.Vector.unit(3, di, int))
                for I in ti.grouped(self.tmp1):
                    r = tmp[di][I] * self.gwei[0]
                    for i in range(1, self.radius + 1):
                        r += (tmp[di][I + dir * i] + tmp[di][I - dir * i]) * self.gwei[i]
                    if ti.static(di == 3):
                        if r <= 0.1:
                            continue
                    tmp[di + 1][I] = r


_et2 = np.array([
        [[-1, -1], [-1, -1]],  #
        [[0, 1], [-1, -1]],  #a
        [[0, 2], [-1, -1]],  #b
        [[1, 2], [-1, -1]],  #ab
        [[1, 3], [-1, -1]],  #c
        [[0, 3], [-1, -1]],  #ca
        [[2, 3], [0, 1]],  #cb
        [[2, 3], [-1, -1]],  #cab
        [[2, 3], [-1, -1]],  #d
        [[2, 3], [0, 1]],  #da
        [[0, 3], [-1, -1]],  #db
        [[1, 3], [-1, -1]],  #dab
        [[1, 2], [-1, -1]],  #dc
        [[0, 2], [-1, -1]],  #dca
        [[0, 1], [-1, -1]],  #dcb
        [[-1, -1], [-1, -1]],  #dcab
    ], np.int32)

# https://www.cnblogs.com/shushen/p/5542131.html
_et3b = b'|NsC0|NsC0|NsC0|Ns902m}BB|NsC0|NsC0|Nj613IG59|NsC0|NsC0{{aXC2?zoI|NsC0|NsC00RjsD|NsC0|NsC0|Ns902m=8E3jhEA|NsC0|NjXB3IGBL|NsC0|NsC0{{jdD0tyHU2?+oH|NsC00}BHG|NsC0|NsC0|Ns903jzoW0RR90|NsC0|Nj9A00ILG|NsC0|NsC0{{agE0SOBU2n+xJ|NsC00}25P3IqTD|NsC0|Ns903IPBJ3J41d|NsC0|NjFC00RpN3knJU|NsC0{|N{R3J44T|NsC0|NsC01P2KJ|NsC0|NsC0|Ns940{{mD1poj4|NsC0|Nj612?zuS|NsC0|NsC0{{#UE1P1{J0|Ed4|NsC00RjpL1PA~B|NsC0|Ns931P22E1OWmH|NsC0|NjXB3JCxL2m}ZJ|NsC0{{jjL0tp8K2LlHQ1poj42m}WM3j+WD|NsC0|Ns9B1P2QO1OfmA|NsC0|NjX90SE*K0s{;G|NsC0{{#mM2?PrX3jzrO0ssI10}25H3knAa1poj4|Ns913km@Q3jqKG2MYxM|NjIB2nhfS2@47f00aO3{{#mM1PciX3kv`L|NsC02?YfI|NsC0|NsC0|Ns991q1*H1ONa3|NsC0|Nj651OWvA|NsC0|NsC0{|E&H2m=KJ0R{j6|NsC00RjpM1qA>9|NsC0|Ns9300;pB3IquS|NsC0|NjL73IzlL1ONj6|NsC0{{jjH0|EsD1q1^G2><{82?YcK0}KEE|NsC0|Ns903jzQL3j_%T|NsC0|Nj651ONdA0s{;G|NsC0{{jI80tE;H2nz%V1^@s53IhuY0RssI1poj4|Ns942?YQM0SF2K2n!1T|NjL900jUG1q%ub00aO3{{;jH1qccX2n+xJ|NsC02?q!T2MPcG|NsC0|Ns990{{sH0|f^I|NsC0|Nj672mk>G0R;#D|NsC0{{aO90|f{F|NsC0|NsC02?q!X1qTWN0{{R2|Ns9A0RjmH00jd81qTEF|NjU80tf;H2n7cU1p@#7{{jjH0tEvD1qc8C|NsC02MGlS2nhoV0{{R2|Ns991qTTS0to^D0tXBK|NjC53jhHK0S5>H1qc8C{|f>E3jqfQ0R{j6|NsC02?YoU1qTWN0}2BR|Ns952LJ^C2?q-R0RRdM3IGcV01E&E3IzZN00#vJ01FBQ2MY!N|NsC0|NsC03I+xL|NsC0|NsC0|Ns902m=KQ2LJ#6|NsC0|NjX90R;*M|NsC0|NsC0{{aXC0SO2N3I_lG|NsC00R{yE1_A&7|NsC0|Ns911_c2E1_J;H|NsC0|NjXF1qlEK00IX8|NsC0{{;yM1qcEK0tN#D2><{80s{*Q1_l5B|NsC0|Ns9B00;{L015^L|NsC0|Nj612?7HP1qufL|NsC0{{;#L0SN*L3jzrU3;+NB1_KKQ1p@^E1ONa3|Ns902nzrU1poyB1q%lM|NjFE1^@#F00spB1quKE{{{sK1_=ub2?+oH|NsC01qucP2MGWF|NsC0|Ns940{{dE0|o^O|NsC0|Nj9A00jyL2m}ZJ|NsC0{|W{L0SN~I2LlHQ1poj41_1&F1px#H2><{8|Ns910tE#E1_J;D0|W>E|NjUC2MGWL00spB0tWy8{|5sJ2MGiN0tp2P1_A~N0}BEN2m}fS1^@s5|Ns953I+rR0t5m80tXBK|Nj612?PfS0s{*L3I_lG{|N#C2@3)V1PccX1O*BP2m}WM3k3rO0R;;N|Ns950Sg5S1_1yI2MYuM1PcHK2>=EK00RaK1_KBL2L=TR1_=uU2MGrY3IG593IquT1PcHE|NsC0|Ns943I+rT3IGTL|NsC0|NjaA0SX2H1_S{A|NsC0{|EyC2muBN1_TBH3jhEA0R#yF0t5mE1poj4|Ns9300;pB2?7KO0tN*C|Nj621Ox&G|NsC0|NsC0{|EyD2m%BI0tWy8|NsC03IquX1_TQN1ONa3|Ns902m%5K3j_%Y1PTWK|NjFE0ssL900smG0Sf>B{{{pB1_25L2mt~C3kVAV2?hiS0|p5J0}BQN|Ns983jqiL01E~I2>}EK1OWpJ1_K5F00spA|NsC0{{{pI3kC@P|NsC0|NsC02MPuU2nq-Z3jhEA|Ns902Lk{K2LK5Q1_uiN|NjaG2LTEP0S5>H2mt^8{|W{N3I_oJ2Lu2A|NsC00RjdA1_%KN2?z!U|Ns921_=TQ0R{&N00{#J0|^HR00#gD1^@#8|NsC0{|5sC1_uKF|NsC0|NsC00s{*Q1_%lW2?z!U|Ns9200#mG3jhfR1_ufW3I_oQ009RG0SX5O2MPiM3kw1P3jqfT1_1^K0ssI12nhxV1_ucN1`7rQ0RsjA2>}ZR2mk;7|NsC0|NjRF00#gD0}B8P1_1y6{|5^O|NsC0|NsC0|NsC02L=oO|NsC0|NsC0|Ns9300;{Q2LJ#6|NsC0|Nj612@3}X|NsC0|NsC0{|EsI2m=8N2L}KD|NsC03IPHJ3kU!I|NsC0|Ns910ty2F2nGuW|NsC0|NjCB00IgL1`7xO|NsC0{{{;O0ty2P2m=ZU2><{82Lb~I0tf&9|NsC0|Ns9700;*L00sg8|NsC0|NjC91_A>I009aA|NsC0{{aR90SE>G2?z)W2LJ#63I_%X0S5sC2mk;7|Ns9A2L=HL3IPZQ0RRa9|Nj632LJ~O0162P3J3rH{|5#N2MP!X3JL%J|NsC01_%TT2nPTE|NsC0|Ns931`7iK1^@&G|NsC0|NjUE3kU=T2>=2A|NsC0{|N*J2?hfR0|5&I2LJ#61_%TO3kU)V0ssI1|Ns910ty2F3jhWS00ajA|NjIF2m}TT00IdD3JL%J{|X5M3IhTO1Op2L1_TBJ2m%8L1OfyG0{{R2|Ns901OfyG0{{R2|NsC0|Nj9A00IL90t5yG0|@{B{{aaE0R#d91P1^A|NsC02mu2K1_1~J1_lZN|Ns9A0RRdB1_l5G|NsC0|NjIA0|WyI1_}cJ0|^QV0}2TQ1_}iK|NsC0|NsC01PKKP1`GfH|NsC0|Ns902m=HO1q%lT|NsC0|NjL50R;pA2L=oO|NsC0{|g5O2m=HI1q1^D1^@s52?YcS0RjgG3;+NB|Ns963kLxL3IGTL1PKNI|NjRD3k3uU1Of^K00RI2{{sXF0|f*F0tE^M0t*KQ2Lb~J1_A{H3IG59|Ns991q1*H1^@;E1_%fL|NjF90s{vI0R;dB1OWg4{{{jG1_%cN0SE*L1px&J2?YcS0R{mF1_1*H|Ns911_}WO1_1yE2nPTO1q1{D3IqxT00RmJ3I_uR3I_%X2MP!U1PTNQ3jhEA1_=cQ3keGd3IG59|Ns931`7ZN0{{gE00{;E|Nj6B2ml2O009LB1`GfH{{{;K1_K2J0|Ed4|NsC00RjpM1q%ra2nz)U|Ns903j+WK3jhfQ1qKNL0tyQV1q%fR2ml2N1p)v91qKTP1_K2G3IhrS1ONa31qcZR0tf{L0s{yF|Ns991qKNQ000I8|NsC0|Nj962muHH1qKKM2m%HI2mu8K0s#j9|NsC0|NsC00RsjB1_}cR1_cHQ2nhxX0RRdB1_=cK1qJ~B|Nj632n7ZT|NsC0|NsC0{|W^L|NsC0|NsC0|NsC03k3=X1q=WG|NsC0|Ns9B1qurX1qcHG|NsC0|NjLG2L%cX0SN&A|NsC0{|W~M3JV7b2muHK0ssI13jqQP2LT5K0ssI1|Ns902m=8E2LT5K2LcQK|NjXG1qlKN2>=2D3kU!I{|5yE2LcNP2?7HG2nh%R0tE^J0|f&I1^@s5|Ns980ssgF0tg2M3IYZH|NjX90R;*J1p@~I3IhNC{|N{J2?7BK2LcKL1qTHJ0RsgC2L=EC|NsC0|Ns902nPTM0RaaE|NsC0|NjX90|^5K1p^2F|NsC0{|N{O1qlcL|NsC0|NsC01qcKM3J3}d2><{8|Ns9500adK00jyQ3j+ZE|Nj612?zuV2nq`d1O@;9{|XBP3IqiU0|W^K0RsU90tEpA2n7NQ2m}QP|Ns901PcHQ0|W&N0t*2J0Sf>E1poyJ0t*EM1qcfW1qlQN0t*BG|NsC0|NsC00tE^K1p)&E1p^2K|Ns953IYWJ1Ox&A|NsC0|NjFD0s{pK0|*5K1qc8E2?YuQ1p)*C2?7ZO0{{R22m}QP1p@;G0ssI1|Ns901O)*A1^@s5|NsC0|NjUC1qcNL2>=BE0|o#8{|N*I|NsC0|NsC0|NsC01PccQ2@44d3;+NB|Ns902m=HO2MG%Y2?`7U|Nj9B3jqrR0R#XC1PlNF{{sO80|W>G3IqoP3knMa1PccV3j_%Q3kd-N|Ns992LuTV2MGZS0t*2E2m=cT1PcTL0t5j6|NsC0{|g5M3j_iP0|WyC1poj40tpHN2MGcL2L}WR|Ns993I_=X1PTHN2nPTH00#pH3IhrP2LuWM3IGHE3IPfN2nPiJ|NsC0|NsC01PK8I0S5;G1ONa3|Ns942>}EF2LK2G2nPZG|NjI40|x{H|NsC0|NsC0{{#pJ|NsC0|NsC0|NsC02?_`b3kd)J|NsC0|Ns9300{#L3kwMf|NsC0|Nj613IGZS2nq}T|NsC0{{sOE3j+%O|NsC0|NsC00RjsF3keAe2><{8|Ns9300{#L3jqQN0t*TM|Nj623kU!U|NsC0|NsC0{{sRG|NsC0|NsC0|NsC00s{yF2nq@a3IG59|Ns993IYHL0{{R2|NsC0|NjC52m%NS009UA3JCxI{{adD|NsC0|NsC0|NsC00RspL0SN#9|NsC0|Ns902?78A|NsC0|NsC0|Nj632><{8|NsC0|NsC0|NsC0|NsC0|NsC0|NsC0'
_et3 = np.frombuffer(base64.b85decode(_et3b), dtype=np.int8).reshape(256, 5, 3).astype(np.int32)
