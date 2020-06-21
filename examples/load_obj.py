import taichi as ti
import taichi_glsl as ts
import taichi_three as t3
import numpy as np


obj = t3.readobj('examples/torus.obj', scale=0.5)

n_faces = obj['f'].shape[0]
n_vertices = obj['v'].shape[0]


ti.init(ti.cpu)

scene = t3.Scene()

model = t3.Model()
v_pos = ti.Vector.var(3, ti.f32, n_vertices)
f_idx = ti.Vector.var(3, ti.i32, n_faces)
v_pos.from_numpy(obj['v'])
f_idx.from_numpy(obj['f'])
model.add_geometry(t3.Face(f_idx))
model.set_vertices(t3.Vertex(v_pos))

scene.add_model(model)
scene.set_light_dir([0.4, -1.5, -1.8])


gui = ti.GUI('Loading Model', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    scene.camera.from_mouse(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
