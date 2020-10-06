import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model(faces_n=2, pos_n=3, tex_n=1, nrm_n=2)
scene.add_model(model)

camera = t3.Camera()
scene.add_camera(camera)

light = t3.Light([0.4, -1.5, -1.8])
scene.add_light(light)

model.pos[0] = [+0.0, +0.5, 0.0]
model.pos[1] = [-0.5, -0.5, 0.0]
model.pos[2] = [+0.5, -0.5, 0.0]
model.nrm[0] = [0.0, 0.0, -1.0]
model.nrm[1] = [0.0, 0.0, +1.0]
model.faces[0] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
model.faces[1] = [[0, 0, 1], [2, 0, 1], [1, 0, 1]]

gui = ti.GUI('Hello Triangle')
while gui.running:
    scene.render()
    gui.get_event(None)
    camera.from_mouse(gui)
    gui.set_image(camera.img)  # blit the image captured by `camera`
    gui.show()