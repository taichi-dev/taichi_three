import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

mtllib = [None]
scene = t3.Scene()
parts = t3.objunpackmtls(t3.readobj('assets/multimtl.obj', scale=0.8))
model1 = t3.Model(t3.Mesh.from_obj(parts[b'Material1']))
model2 = t3.Model(t3.Mesh.from_obj(parts[b'Material2']))
model1.material = t3.DeferredMaterial(mtllib, t3.Material(t3.CookTorrance(  # Up: gold
    color=t3.Constant(t3.RGB(1.0, 0.96, 0.88)),
    roughness=t3.Constant(0.2),
    metallic=t3.Constant(0.75),
    )))
model2.material = t3.DeferredMaterial(mtllib, t3.Material(t3.CookTorrance(  # Down: cloth
    color=t3.Texture('assets/cloth.jpg'),
    roughness=t3.Constant(0.3),
    metallic=t3.Constant(0.0),
    )))
scene.add_model(model1)
scene.add_model(model2)
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[0.8, 0, 2.5])
scene.add_camera_d(camera)
gbuff = t3.FrameBuffer(camera, buffers=dict(
    mid=[(), int],
    position=[3, float],
    texcoord=[2, float],
    normal=[3, float],
    tangent=[3, float]))
imgbuf = t3.DeferredShading(gbuff, mtllib)
scene.add_buffer(imgbuf)
light = t3.Light([0, -0.5, -1], 0.9)
scene.add_light(light)
ambient = t3.AmbientLight(0.1)
scene.add_light(ambient)

gui = ti.GUI('Model', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    model2.L2W[None] = model1.L2W[None] = t3.rotateX(angle=t3.get_time())
    scene.render()
    gui.set_image(imgbuf.img)
    gui.show()
