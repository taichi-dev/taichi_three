import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
import numpy as np


model = t3.readobj('examples/torus.obj') / 2

N = model.shape[0]


ti.init(ti.cpu)

scene = t3.SceneGE()
pos1 = ti.Vector(3, ti.f32)
pos2 = ti.Vector(3, ti.f32)
pos3 = ti.Vector(3, ti.f32)
ti.root.dense(ti.i, N).place(pos1, pos2, pos3)


scene.add_triangle(pos1, pos2, pos3)
scene.set_light_dir([0.4, -1.5, -1.8])
pos1.from_numpy(model[:, 0])
pos2.from_numpy(model[:, 1])
pos3.from_numpy(model[:, 2])


gui = ti.GUI('Loading Model', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    scene.camera.from_mouse(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
