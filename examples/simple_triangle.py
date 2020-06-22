import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model()
scene.add_model(model)

vertices = t3.Vertex.var(3)
faces = t3.Face.var(2)
vertices.pos.from_numpy(np.array(
    [[+0.0, +0.5, 0.0], [-0.5, -0.5, 0.0], [+0.5, -0.5, 0.0]]))
faces.idx.from_numpy(np.array([[0, 1, 2], [0, 2, 1]]))
model.set_vertices(vertices)
model.add_geometry(faces)

scene.set_light_dir([0.4, -1.5, -1.8])
gui = ti.GUI('Triangle', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    #model.L2W.from_mouse(gui)
    scene.camera.from_mouse(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
