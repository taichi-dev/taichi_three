import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
obj1 = t3.readobj('assets/torus.obj', scale=0.8)
obj2 = t3.readobj('assets/cylinder.obj', scale=0.6)
model1 = t3.ModelPP.from_obj(obj1)
model2 = t3.ModelPP.from_obj(obj2)
scene.add_model(model1)
scene.add_model(model2)
camera = t3.Camera(pos=[0, 1, -1.8])
scene.add_camera(camera)
light = t3.Light([0.4, -1.5, 1.8])
shadow = t3.Camera(pos=[-0.4*2, 1.5*2, -1.8*2], res=(512, 512), fov=50)
shadow.type = shadow.ORTHO
light.bind_shadow(shadow)
scene.add_shadow_camera(shadow)
scene.add_light(light)

gui = ti.GUI('Model', camera.res)
gui2 = ti.GUI('Depth map', shadow.res)
gui2.fps_limit = None
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    model2.L2W.offset[None] = [0, 0.16 * ti.sin(gui.frame * 0.03), 0]
    scene.render()
    gui.set_image(camera.img)
    gui2.set_image(shadow.fb['idepth'])
    gui.show()
    gui2.show()
