import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
skybox = t3.Skybox('assets/skybox.jpg')
scene.add_model(skybox)
scene.add_light(skybox)
model = t3.Model(t3.Mesh.from_obj('assets/sphere.obj'))
model.material = t3.Material(t3.IdealRT(
    specular=t3.Constant(1.0),
))
scene.add_model(model)
camera = t3.Camera(res=(600, 400))
camera.ctl = t3.CameraCtl(pos=[0, 0, 3])
scene.add_camera(camera)

gui = ti.GUI('Skybox', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
