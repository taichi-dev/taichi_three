from ..advans import *


@ti.func
def ray_aabb_hit(bmin, bmax, ro, rd):
    near = -inf
    far = inf
    hit = 1

    for i in ti.static(range(bmin.n)):
        if abs(rd[i]) < eps:
            if ro[i] < bmin[i] or ro[i] > bmax[i]:
                hit = 0
        else:
            i1 = (bmin[i] - ro[i]) / rd[i]
            i2 = (bmax[i] - ro[i]) / rd[i]

            far = min(far, max(i1, i2))
            near = max(near, min(i1, i2))

    if near > far:
        hit = 0

    return hit, near


@ti.func
def ray_triangle_hit(self, v0, v1, v2, ro, rd):
    e1 = v1 - v0
    e2 = v2 - v0
    p = rd.cross(e2)
    det = e1.dot(p)
    s = ro - v0

    t, u, v = inf, 0.0, 0.0
    ipos = ti.Vector.zero(float, 3)
    inrm = ti.Vector.zero(float, 3)
    itex = ti.Vector.zero(float, 2)

    if det < 0:
        s = -s
        det = -det

    if det >= eps:
        u = s.dot(p)
        if 0 <= u <= det:
            q = s.cross(e1)
            v = rd.dot(q)
            if v >= 0 and u + v <= det:
                t = e2.dot(q)
                det = 1 / det
                t *= det
                u *= det
                v *= det
                inrm = e2.cross(e1).normalized()
                ipos = ro + t * rd
                itex = V(u, v)

    return t, ipos, inrm, itex


@ti.func
def ray_sphere_hit(pos, rad, ro, rd):
    t = inf * 2
    op = pos - ro
    b = op.dot(rd)
    det = b**2 - op.norm_sqr() + rad**2
    if det >= 0:
        det = ti.sqrt(det)
        t = b - det
        if t <= eps:
            t = b + det
            if t <= eps:
                t = inf * 2
    return t < inf, t
