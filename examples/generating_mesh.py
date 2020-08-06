import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
ti.init(ti.opengl, kernel_profiler=True)

scene = t3.Scene()
model = t3.Model()
scene.add_model(model)

N = 2 ** 12
faces = t3.Face.var()
vertices = t3.Vertex.var()
#vertices = ti.Vector.var(3, ti.f32, None)
#faces = ti.Vector.var(3, ti.i32, None)
ti.root.dense(ti.i, N * 3).place(vertices)
#print(vertices)
ti.root.dense(ti.i, N).place(faces)
vertices_len = ti.var(ti.i32, ())
faces_len = ti.var(ti.i32, ())

model.set_vertices(vertices)
model.add_geometry(faces)

@ti.func
def glVertex(pos):
    l = ti.atomic_add(vertices_len[None], 1)
    vertices.pos[l] = pos
    return l

@ti.func
def glFace(idx):
    l = ti.atomic_add(faces_len[None], 1)
    faces.idx[l] = idx
    return l

@ti.func
def glTri(A, B, C):
    i = glVertex(A)
    j = glVertex(B)
    k = glVertex(C)
    glFace(ts.vec3(i, j, k))

@ti.func
def glQuad(A, B, C, D):
    glTri(A, B, C)
    glTri(C, D, A)

@ti.func
def glCircle(center, dir1, dir2, N: ti.template()):
    for n in range(N):
        a = n * ts.math.tau / N
        b = a - ts.math.tau / N
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
        glQuad(p, q, q + polar, p + polar)

    glCircle(center, dir2, dir1, N)
    glCircle(center + polar, dir1, dir2, N)

@ti.func
def glSphere(center, polar, dir1, dir2, M: ti.template(), N: ti.template()):
    for m, n in ti.ndrange((1 - M, M - 1), N):
        a = n * ts.math.tau / N
        b = a + ts.math.tau / N
        c = m * ts.math.pi / 2 / M
        d = c + ts.math.pi / 2 / M
        r, h = ti.cos(c), ti.sin(c)
        r1, r2 = r * dir1, r * dir2
        p1 = center + polar * h + r1 * ti.cos(a) + r2 * ti.sin(a)
        q1 = center + polar * h + r1 * ti.cos(b) + r2 * ti.sin(b)
        r, h = ti.cos(d), ti.sin(d)
        r1, r2 = r * dir1, r * dir2
        p2 = center + polar * h + r1 * ti.cos(a) + r2 * ti.sin(a)
        q2 = center + polar * h + r1 * ti.cos(b) + r2 * ti.sin(b)
        glQuad(q1, p1, p2, q2)
    r, h = ti.sin(ts.math.pi / 2 / M), ti.cos(ts.math.pi / 2 / M)
    r1, r2 = r * dir1, r * dir2
    cp1 = center + polar
    cph = center + polar * h
    cq1 = center - polar
    cqh = center - polar * h
    for n in range(N):
        a = n * ts.math.tau / N
        b = a - ts.math.tau / N
        p = cph + r1 * ti.cos(a) + r2 * ti.sin(a)
        q = cph + r1 * ti.cos(b) + r2 * ti.sin(b)
        glTri(p, q, cp1)
        p = cqh + r1 * ti.cos(a) + r2 * ti.sin(a)
        q = cqh + r1 * ti.cos(b) + r2 * ti.sin(b)
        glTri(q, p, cq1)


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
               64)


@ti.kernel
def initSphere():
    glSphere(ts.vec3(0.0, 0.0, 0.0),
             ts.vec3(0.0, 0.0, 0.6),
             ts.vec3(0.5, 0.0, 0.0),
             ts.vec3(0.0, 0.5, 0.0),
             16, 64)


initSphere()

scene.set_light_dir([1, 1, -1])
gui = ti.GUI('Mesh of Triangles', scene.res)
while gui.running:
    gui.get_event()
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    model.L2W.from_mouse(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()

ti.kernel_profiler_print()
