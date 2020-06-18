import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
ti.init(ti.opengl)

scene = t3.SceneGE()
pos1 = ti.Vector(3, ti.f32)
pos2 = ti.Vector(3, ti.f32)
pos3 = ti.Vector(3, ti.f32)
ti.root.dense(ti.i, 1).place(pos1, pos2, pos3)

pos1[0] = [-0.5, +0.5, 0.0]
pos2[0] = [+0.5, +0.5, 0.0]
pos3[0] = [-0.5, -0.5, 0.0]

scene.add_triangle(pos1, pos2, pos3)
scene.set_light_dir([1, 1, -1])

gui = ti.GUI('Triangle', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    scene.camera.from_mouse(gui, dis=4)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
