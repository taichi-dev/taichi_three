import taichi as ti
import taichi_three as t3
import numpy as np

res = 512, 512
ti.init(ti.cpu)

scene = t3.Scene()
cornell = t3.objunpackmtls(t3.readobj('assets/cornell.obj'))
plane = t3.readobj('assets/plane.obj')
model1 = t3.Model(t3.Mesh.from_obj(cornell[b'Material']))
model1.material = t3.Material(t3.IdealRT(
    specular=t3.Constant(0.0),
    diffuse=t3.Constant(1.0),
    emission=t3.Constant(0.0),
    diffuse_color=t3.Texture('assets/smallptwall.png'),
))
scene.add_model(model1)
model2 = t3.Model(t3.Mesh.from_obj(cornell[b'Material.001']))
model2.material = t3.Material(t3.IdealRT(
    specular=t3.Constant(0.7),
    diffuse=t3.Constant(1.0),
    emission=t3.Constant(0.0),
))
scene.add_model(model2)
light = t3.Model(t3.Mesh.from_obj(plane))
light.material = t3.Material(t3.IdealRT(
    specular=t3.Constant(0.0),
    diffuse=t3.Constant(0.0),
    emission=t3.Constant(1.0),
    emission_color=t3.Constant(16.0),
))
scene.add_model(light)
camera = t3.RTCamera(res=res)
camera.ctl = t3.CameraCtl(pos=[0, 2, 8], target=[0, 2, 0])
scene.add_camera_d(camera)
camfb = t3.FrameBuffer(camera)
if isinstance(camera, t3.RTCamera):
    camfb.clear_buffer = lambda: None
else:
    scene.add_light(t3.PointLight(pos=(0, 3.9, 0), color=6.0))
accum = t3.AccDenoise(camfb)
buffer = t3.ImgUnaryOp(accum, lambda x: 1 - ti.exp(-x))
scene.add_buffer(buffer)

light.L2W[None] = t3.translate(0, 3.9, 0) @ t3.scale(0.25)
gui = ti.GUI('Path tracing', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    if camera.from_mouse(gui):
        accum.reset()
    if isinstance(camera, t3.RTCamera):
        camera.loadrays()
        for i in range(3):
            camera.steprays()
        camera.applyrays()
    scene.render()
    gui.set_image(buffer.img)
    gui.show()
