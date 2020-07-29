import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model(f_n=2, vi_n=3)
scene.add_model(model)

camera = t3.Camera()
scene.add_camera(camera)

light = t3.Light([0.4, -1.5, 1.8])
scene.add_light(light)

model.vi.from_numpy(np.array(
    [[+0.0, +0.5, 0.0], [-0.5, -0.5, 0.0], [+0.5, -0.5, 0.0]]))
model.faces.from_numpy(np.array([[0, 1, 2], [0, 2, 1]])) # both cull faces

gui = ti.GUI('Triangle', camera.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
