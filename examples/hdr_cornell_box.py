import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
cornell = t3.readobj('assets/cornell.obj')
cube = t3.readobj('assets/plane.obj')
model = t3.Model(t3.Mesh.from_obj(cornell))
scene.add_model(model)
light = t3.PointLight(pos=[0.0, 3.9, 0.0], color=15.0)
scene.add_light(light)
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[0, 2, 6], target=[0, 2, 0])
scene.add_camera_d(camera)
original = t3.FrameBuffer(camera)
mapped = t3.ImgUnaryOp(original, lambda x: 1 - ti.exp(-x))
scene.add_buffer(mapped)

#light.L2W[None] = t3.translate(0, 3.9, 0)
gui_ldr = ti.GUI('LDR', camera.res)
gui_hdr = ti.GUI('HDR', camera.res)
while gui_ldr.running and gui_hdr.running:
    gui_hdr.get_event(None)
    camera.from_mouse(gui_hdr)
    scene.render()
    gui_ldr.set_image(original.img)
    gui_ldr.show()
    gui_hdr.set_image(mapped.img)
    gui_hdr.show()
