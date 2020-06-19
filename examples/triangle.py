import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
ti.init(ti.cpu)

scene = t3.SceneGE()
pos1 = ti.Vector(3, ti.f32)
pos2 = ti.Vector(3, ti.f32)
pos3 = ti.Vector(3, ti.f32)
tris = ti.root.dynamic(ti.i, 1024)
tris.place(pos1, pos2, pos3)
tris_len = ti.var(ti.i32, ())

scene.add_triangle(pos1, pos2, pos3)
scene.set_light_dir([1, 1, -1])

@ti.func
def glTri(A, B, C):
    l = ti.atomic_add(tris_len[None], 1)
    pos1[l] = A
    pos2[l] = B
    pos3[l] = C

@ti.func
def glQuad(A, B, C, D):
    glTri(A, B, C)
    glTri(C, D, A)

@ti.func
def glCircle(center, dir1, dir2, N: ti.template()):
    for n in range(N):
        a = n * ts.math.tau / N
        b = a + ts.math.tau / N
        p = center + dir1 * ti.cos(a) + dir2 * ti.sin(a)
        q = center + dir1 * ti.cos(b) + dir2 * ti.sin(b)
        glTri(p, center, q)

@ti.func
def glCylinder(center, polar, dir1, dir2, N: ti.template()):
    for n in range(N):
        a = n * ts.math.tau / N
        b = a + ts.math.tau / N
        p = center + dir1 * ti.cos(a) + dir2 * ti.sin(a)
        q = center + dir1 * ti.cos(b) + dir2 * ti.sin(b)
        glQuad(p, q, p + polar, q + polar)

    glCircle(center, dir1, dir2, N)
    glCircle(center + polar, dir1, dir2, N)

@ti.func
def glSphere(center, polar, dir1, dir2, M: ti.template(), N: ti.template()):
    for m, n in ti.ndrange((-M, M + 1), N):
        a = n * ts.math.tau / N
        b = a + ts.math.tau / N
        c = m * ts.math.pi / 2 / M
        d = c - ts.math.pi / 2 / M
        r, h = ti.cos(c), ti.sin(c)
        r1, r2 = r * dir1, r * dir2
        p1 = center + polar * h + r1 * ti.cos(a) + r2 * ti.sin(a)
        q1 = center + polar * h + r1 * ti.cos(b) + r2 * ti.sin(b)
        r, h = ti.cos(d), ti.sin(d)
        r1, r2 = r * dir1, r * dir2
        p2 = center + polar * h + r1 * ti.cos(b) + r2 * ti.sin(b)
        q2 = center + polar * h + r1 * ti.cos(b) + r2 * ti.sin(b)
        glQuad(p1, q1, p2, q2)


@ti.kernel
def initQuad():
    glQuad(
            ts.vec3(-0.5, -0.5, 0.0),
            ts.vec3(+0.5, -0.5, 0.0),
            ts.vec3(+0.5, +0.5, 0.0),
            ts.vec3(-0.5, +0.5, 0.0))


@ti.kernel
def initCircle():
    glCircle(ts.vec3(0.0, 0.0, 0.0),
            ts.vec3(0.5, 0.0, 0.0),
            ts.vec3(0.0, 0.5, 0.0),
            32)


@ti.kernel
def initCylinder():
    glCylinder(ts.vec3(0.0, 0.0,-0.5),
               ts.vec3(0.0, 0.0, 1.0),
               ts.vec3(0.5, 0.0, 0.0),
               ts.vec3(0.0, 0.5, 0.0),
               16)


@ti.kernel
def initSphere():
    glSphere(ts.vec3(0.0, 0.0, 0.0),
             ts.vec3(0.0, 0.0, 0.5),
             ts.vec3(0.5, 0.0, 0.0),
             ts.vec3(0.0, 0.5, 0.0),
             8, 16)


initSphere()

gui = ti.GUI('Mesh of Triangles', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    scene.camera.from_mouse(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
