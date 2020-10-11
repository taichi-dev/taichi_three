import taichi as ti
import taichi_three as t3
import numpy as np

res = 512, 512
ti.init(ti.cpu)

scene = t3.Scene()
obj = t3.readobj('assets/cube.obj', scale=0.8)
model = t3.Model.from_obj(obj)
model.shading_type = t3.IdealRT
model.add_uniform('diffuse', 1.0)
model.add_uniform('emission', 0.0)
scene.add_model(model)
light = t3.Model.from_obj(obj)
light.shading_type = t3.IdealRT
light.add_uniform('diffuse', 0.0)
light.add_uniform('emission', 1.0)
light.add_uniform('emission_color', 16.0)
scene.add_model(light)
camera = t3.RTCamera(res=res)
camera.ctl = t3.CameraCtl(pos=[0.4, 0, -3.7])
scene.add_camera(camera)
accumator = t3.Accumator(camera.res)

light.L2W[None] = t3.translate(0, 0, -1.6) @ t3.scale(0.2)
gui = ti.GUI('Model', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    if camera.from_mouse(gui):
        accumator.reset()
    accumator.render(camera, 2)
    gui.set_image(accumator.buf)
    gui.show()
