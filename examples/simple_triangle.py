import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model(faces_n=2, pos_n=3, tex_n=1, nrm_n=1)
scene.add_model(model)

camera = t3.Camera()
scene.add_camera(camera)

light = t3.Light([0.4, -1.5, 1.8])
scene.add_light(light)

model.pos.from_numpy(np.array(
    [[+0.0, +0.5, 0.0], [-0.5, -0.5, 0.0], [+0.5, -0.5, 0.0]]))
model.nrm.from_numpy(np.array(
    [[0.0, 0.0, -1.0], [0.0, 0.0, +1.0]]))
model.faces.from_numpy(np.array(
    [[[0, 1, 2], [0, 0, 0], [0, 0, 0]],
     [[0, 1, 2], [0, 0, 0], [1, 1, 1]]]))

gui = ti.GUI('Triangle', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
