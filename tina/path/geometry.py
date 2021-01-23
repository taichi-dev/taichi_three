from ..advans import *


@ti.func
def ray_aabb_hit(bmin, bmax, ro, rd):
    near = -inf
    far = inf

    for i in ti.static(range(bmin.n)):
        if abs(rd[i]) < eps:
            if ro[i] < bmin[i] or ro[i] > bmax[i]:
                hit = 0
        else:
            i1 = (bmin[i] - ro[i]) / rd[i]
            i2 = (bmax[i] - ro[i]) / rd[i]

            far = min(far, max(i1, i2))
            near = max(near, min(i1, i2))

    return near, far

'''
@ti.func
def ray_triangle_hit(v0, v1, v2, ro, rd):
    e1 = v1 - v0
    e2 = v2 - v0
    p = rd.cross(e2)
    det = e1.dot(p)
    s = ro - v0

    t, u, v = inf * 2, 0.0, 0.0

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

    return t, V(u, v)
'''

#'''
@ti.func
def ray_triangle_hit(v0, v1, v2, ro, rd):
    u = v1 - v0
    v = v2 - v0
    norm = u.cross(v)
    depth = inf * 2
    s, t = 0., 0.
    hit = 0

    b = norm.dot(rd)
    if abs(b) >= eps:
        w0 = ro - v0
        a = -norm.dot(w0)
        r = a / b
        if r > 0:
            ip = ro + r * rd
            uu = u.dot(u)
            uv = u.dot(v)
            vv = v.dot(v)
            w = ip - v0
            wu = w.dot(u)
            wv = w.dot(v)
            D = uv * uv - uu * vv
            s = (uv * wv - vv * wu) / D
            t = (uv * wu - uu * wv) / D
            if 0 <= s <= 1:
                if 0 <= t and s + t <= 1:
                    depth = r
                    hit = 1
    return hit, depth, V(s, t)


@ti.func
def ray_triangle_cull_hit(v0, v1, v2, ro, rd):
    u = v1 - v0
    v = v2 - v0
    norm = u.cross(v)
    depth = inf * 2
    s, t = 0., 0.
    hit = 0

    b = -norm.dot(rd)
    if b >= eps:
        w0 = ro - v0
        a = norm.dot(w0)
        r = a / b
        if r > 0:
            ip = ro + r * rd
            uu = u.dot(u)
            uv = u.dot(v)
            vv = v.dot(v)
            w = ip - v0
            wu = w.dot(u)
            wv = w.dot(v)
            D = uv * uv - uu * vv
            s = (uv * wv - vv * wu) / D
            t = (uv * wu - uu * wv) / D
            if 0 <= s <= 1:
                if 0 <= t and s + t <= 1:
                    depth = r
                    hit = 1
    return hit, depth, V(s, t)
#'''

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
