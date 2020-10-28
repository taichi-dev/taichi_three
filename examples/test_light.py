import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model(t3.Mesh.from_obj('assets/torus.obj'))
scene.add_model(model)
orient = t3.Model(t3.Mesh.from_obj(t3.readobj('assets/orient.obj', scale=0.8)))
scene.add_model(orient)
ambient = t3.AmbientLight(0.3)
scene.add_light(ambient)
light = t3.Light([0.4, -1.5, -1.8], 0.7)
scene.add_light(light)
shadow = light.make_shadow_camera()
scene.add_camera(shadow)
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[0, 1, 1.8])
scene.add_camera(camera)

gui = ti.GUI('Camera', camera.res)
gui2 = ti.GUI('Shadow', shadow.res)
gui2.fps_limit = None
while gui.running and gui2.running:
    gui.get_event(None)
    gui2.get_event(None)
    orient.L2W[None] = t3.translate(0, t3.sin(t3.get_time()), 0)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    gui2.running = not gui2.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui2.set_image(shadow.fb.idepth)
    gui.show()
    gui2.show()
