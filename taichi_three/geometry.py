import taichi as ti
import taichi_glsl as ts


@ti.func
def render_triangle(model, face):
    scene = model.scene
    L2W = model.L2W
    ia, ib, ic = model.vi[face[0, 0]], model.vi[face[0, 1]], model.vi[face[0, 2]]
    ta, tb, tc = model.vt[face[1, 0]], model.vt[face[1, 1]], model.vt[face[1, 2]]
    na, nb, nc = model.vn[face[2, 0]], model.vn[face[2, 1]], model.vn[face[2, 2]]
    a = scene.camera.untrans_pos(L2W @ ia)
    b = scene.camera.untrans_pos(L2W @ ib)
    c = scene.camera.untrans_pos(L2W @ ic)
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
    center_pos = (a + b + c) / 3

    color = scene.opt.render_func(center_pos, normal, ts.vec3(0.0), light_dir)
    color = scene.opt.pre_process(color)

    Ak = 1 / (ts.cross(A, C_B) + CxB)
    Bk = 1 / (ts.cross(B, A_C) + AxC)
    Ck = 1 / (ts.cross(C, B_A) + BxA)

    W = 1
    M = int(ti.floor(min(A, B, C) - W))
    N = int(ti.ceil(max(A, B, C) + W))
    for X in ti.grouped(ti.ndrange((M.x, N.x), (M.y, N.y))):
        AB = ts.cross(X, B_A) + BxA
        BC = ts.cross(X, C_B) + CxB
        CA = ts.cross(X, A_C) + AxC
        if AB <= 0 and BC <= 0 and CA <= 0:
            zindex = a.z * Ak * BC + b.z * Bk * CA + c.z * Ck * AB

            if zindex >= ti.atomic_max(scene.zbuf[X], zindex):
                clr = color
                coor = ta * Ak * BC + tb * Bk * CA + tc * Ck * AB
                clr = clr * model.texSample(coor)

                scene.img[X] = clr
