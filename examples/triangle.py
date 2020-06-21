import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
ti.init(ti.cpu)

scene = t3.SceneGE()
pos1 = ti.Vector(3, ti.f32)
pos2 = ti.Vector(3, ti.f32)
pos3 = ti.Vector(3, ti.f32)
tris = ti.root.dynamic(ti.i, 2 ** 12)
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

@ti.kernel
def init():
    glTri(ts.vec3(+0.0, +0.5, 0.0),
          ts.vec3(-0.5, -0.5, 0.0),
          ts.vec3(+0.5, -0.5, 0.0))

init()

gui = ti.GUI('Triangle', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    scene.camera.from_mouse(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
