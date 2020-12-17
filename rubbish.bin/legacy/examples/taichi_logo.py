import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
obj = t3.readobj('assets/logo.obj', scale=0.8)
t3.objbothface(obj)
logo1 = t3.Model(t3.Mesh.from_obj(obj))
logo2 = t3.Model(t3.Mesh.from_obj(obj))
scene.add_model(logo1)
scene.add_model(logo2)
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[-1, 1, 1])
scene.add_camera(camera)
light = t3.Light([0.4, -1.5, 0.8])
scene.add_light(light)

gui = ti.GUI('Taichi THREE', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    t = t3.get_time()
    logo1.L2W[None] = t3.rotateY(t)
    logo2.L2W[None] = t3.rotateX(t3.pi - t) @ t3.rotateZ(t3.pi / 2)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
