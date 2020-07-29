import taichi as ti
import taichi_glsl as ts


@ti.func
def render_triangle(model, camera, face):
    scene = model.scene
    L2W = model.L2W
    _1 = ti.static(min(1, model.faces.m - 1))
    _2 = ti.static(min(2, model.faces.m - 1))
    ia, ib, ic = model.vi[face[0, 0]], model.vi[face[1, 0]], model.vi[face[2, 0]]
    ta, tb, tc = model.vt[face[0, _1]], model.vt[face[1, _1]], model.vt[face[2, _1]]
    na, nb, nc = model.vn[face[0, _2]], model.vn[face[1, _2]], model.vn[face[2, _2]]
    a = camera.untrans_pos(L2W @ ia)
    b = camera.untrans_pos(L2W @ ib)
    c = camera.untrans_pos(L2W @ ic)
    A = camera.uncook(a)
    B = camera.uncook(b)
    C = camera.uncook(c)
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
    color = ts.vec3(0.0)
    center_pos = (a + b + c) / 3

    for light in ti.static(scene.lights):
        light_color = scene.opt.render_func(center_pos, normal, ts.vec3(0.0), light, camera)
        color += light_color

    Ak = 1 / (ts.cross(A, C_B) + CxB)
    Bk = 1 / (ts.cross(B, A_C) + AxC)
    Ck = 1 / (ts.cross(C, B_A) + BxA)

    W = 1
    M = int(ti.floor(min(A, B, C) - W))
    N = int(ti.ceil(max(A, B, C) + W))

    min_z = min(a.z, b.z, c.z)
    if min_z > 0.5:
        for X in ti.grouped(ti.ndrange((max(0, M.x), min(camera.res[0], N.x)), (max(0, M.y), min(camera.res[1], N.y)))):
            AB = ts.cross(X, B_A) + BxA
            BC = ts.cross(X, C_B) + CxB
            CA = ts.cross(X, A_C) + AxC
            if AB >= 0 and BC >= 0 and CA >= 0:
                w_A = max(Ak * BC, 1e-6)
                w_B = max(Bk * CA, 1e-6)
                w_C = max(Ck * AB, 1e-6)
                w_sum = w_A + w_B + w_C
                zindex = 1.0 /  ( (a.z * w_A + b.z * w_B + c.z * w_C) / w_sum)
                if zindex >= ti.atomic_max(camera.zbuf[X], zindex):
                    clr = color
                    coor = (ta * w_A + tb * w_B + tc * w_C) / w_sum
                    clr = clr * model.texSample(coor)

                    camera.img[X] = clr
