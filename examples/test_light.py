import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model(t3.Mesh.from_obj('assets/torus.obj'))
scene.add_model(model)
camera = t3.Camera(res=(600, 400))
camera.ctl = t3.CameraCtl(pos=[0, 1, 1.8])
scene.add_camera(camera)
light = t3.Light([0.4, -1.5, -1.8], 0.7)
scene.add_light(light)
ambient = t3.AmbientLight(0.3)
scene.add_light(ambient)
orient = t3.Model(t3.Mesh.from_obj(t3.readobj('assets/orient.obj', scale=0.4)))
scene.add_model(orient)

orient.L2W[None] = light.L2W[None].value
gui = ti.GUI('Camera', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
