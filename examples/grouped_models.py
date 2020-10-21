import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
model1 = t3.Model(t3.Mesh.from_obj(t3.readobj('assets/torus.obj')))
model2 = t3.Model(t3.Mesh.from_obj(t3.readobj('assets/cube.obj', scale=0.5)))
model = t3.ModelGroup([model1, model2])
scene.add_model(model)
camera = t3.Camera(res=(600, 400))
scene.add_camera(camera)
light = t3.Light([0.4, -1.5, -1.8])
scene.add_light(light)

gui = ti.GUI('Camera', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    model2.L2W[None] = t3.translate(0, ti.sin(t3.get_time()) * 0.6, 0)
    model.L2W[None] = t3.rotateZ(t3.get_time())
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
