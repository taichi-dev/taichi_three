import math
import taichi as ti
import taichi_glsl as ts


@ti.func
def render_triangle(model, camera, face):
    scene = model.scene
    L2W = model.L2W
    posa, posb, posc = model.pos[face[0, 0]], model.pos[face[1, 0]], model.pos[face[2, 0]]
    texa, texb, texc = model.tex[face[0, 1]], model.tex[face[1, 1]], model.tex[face[2, 1]]
    nrma, nrmb, nrmc = model.nrm[face[0, 2]], model.nrm[face[1, 2]], model.nrm[face[2, 2]]
    posa = camera.untrans_pos(L2W @ posa)
    posb = camera.untrans_pos(L2W @ posb)
    posc = camera.untrans_pos(L2W @ posc)
    nrma = camera.untrans_dir(L2W.matrix @ nrma)
    nrmb = camera.untrans_dir(L2W.matrix @ nrmb)
    nrmc = camera.untrans_dir(L2W.matrix @ nrmc)

    # NOTE: the normal computation indicates that # a front-facing face should
    # be COUNTER-CLOCKWISE, i.e., glFrontFace(GL_CCW);
    # this is to be compatible with obj model loading.
    pos_center = (posa + posb + posc) / 3
    if ti.static(camera.type == camera.ORTHO):
        pos_center = ts.vec3(0.0, 0.0, 1.0)
    normal = ts.cross(posa - posb, posa - posc)
    if ts.dot(pos_center, normal) <= 0:
        # shading

        clra = model.vertex_shader(posa, nrma)
        clrb = model.vertex_shader(posb, nrmb)
        clrc = model.vertex_shader(posc, nrmc)

        A = camera.uncook(posa)
        B = camera.uncook(posb)
        C = camera.uncook(posc)
        scr_norm = 1 / ts.cross(A - C, B - A)
        B_A = (B - A) * scr_norm
        C_B = (C - B) * scr_norm
        A_C = (A - C) * scr_norm

        W = 1
        # screen space bounding box
        M = int(ti.floor(min(A, B, C) - W))
        N = int(ti.ceil(max(A, B, C) + W))
        M = ts.clamp(M, 0, ti.Vector(camera.img.shape))
        N = ts.clamp(N, 0, ti.Vector(camera.img.shape))
        for X in ti.grouped(ti.ndrange((M.x, N.x), (M.y, N.y))):
            # barycentric coordinates using the area method
            X_A = X - A
            w_C = ts.cross(B_A, X_A)
            w_B = ts.cross(A_C, X_A)
            w_A = 1 - w_C - w_B
            # draw
            eps = ti.get_rel_eps() * 0.1
            is_inside = w_A >= -eps and w_B >= -eps and w_C >= -eps
            if not is_inside:
                continue
            zindex = 1 / (posa.z * w_A + posb.z * w_B + posc.z * w_C)
            if zindex < ti.atomic_max(camera.zbuf[X], zindex):
                continue

            clr = (clra * w_A + clrb * w_B + clrc * w_C)
            tex = (texa * w_A + texb * w_B + texc * w_C)
            camera.img[X] = model.pixel_shader(clr, tex)


@ti.func
def render_particle(model, camera, vertex, radius):
    scene = model.scene
    L2W = model.L2W
    a = camera.untrans_pos(L2W @ vertex)
    A = camera.uncook(a)

    M = int(ti.floor(A - radius))
    N = int(ti.ceil(A + radius))

    for X in ti.grouped(ti.ndrange((M.x, N.x), (M.y, N.y))):
        if X.x < 0 or X.x >= camera.res[0] or X.y < 0 or X.y >= camera.res[1]:
            continue
        if (X - A).norm_sqr() > radius**2:
            continue

        camera.img[X] = ts.vec3(1)
