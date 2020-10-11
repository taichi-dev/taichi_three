import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
obj1 = t3.readobj('assets/torus.obj', scale=0.8)
obj2 = t3.readobj('assets/cylinder.obj', scale=0.6)
model1 = t3.Model.from_obj(obj1)
model2 = t3.Model.from_obj(obj2)
scene.add_model(model1)
scene.add_model(model2)
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[1, 1, -1])
scene.add_camera(camera)
light = t3.Light([0.4, -1.5, 1.8])
scene.add_shadow_camera(light.make_shadow_camera())
scene.add_light(light)

gui = ti.GUI('Model', camera.res)
gui2 = ti.GUI('Depth map', light.shadow.res)
gui2.fps_limit = None
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    model2.L2W[None] = t3.translate(0, 0.16 * ti.sin(gui.frame * 0.03), 0)
    scene.render_shadows()
    scene.render()
    gui.set_image(camera.img)
    gui2.set_image(light.shadow.img)
    gui.show()
    gui2.show()
