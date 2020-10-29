import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

mtllib = [None]
scene = t3.Scene()
obj = t3.readobj('assets/torus.obj', scale=0.8)
model = t3.Model(t3.Mesh.from_obj(obj))
model.material = t3.DeferredMaterial(mtllib, t3.Material(t3.CookTorrance()))
scene.add_model(model)
camera = t3.Camera(res=(600, 400))
camera.ctl = t3.CameraCtl(pos=[0, 1, -1.8])
scene.add_camera_d(camera)
gbuff = t3.FrameBuffer(camera, buffers=dict(
    mid=[(), int],
    position=[3, float],
    texcoord=[2, float],
    normal=[3, float],
    tangent=[3, float],
))
imgbuf = t3.DeferredShading(gbuff, mtllib)
scene.add_buffer(imgbuf)
light = t3.Light([0.4, -1.5, -0.8], 0.9)
scene.add_light(light)
ambient = t3.AmbientLight(0.1)
scene.add_light(ambient)

gui = ti.GUI('G-Buffer', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(imgbuf.img)
    coor = gui.get_cursor_pos()
    pos = gbuff.fetchpixelinfo('position', coor)
    texcoor = gbuff.fetchpixelinfo('texcoord', coor)
    normal = gbuff.fetchpixelinfo('normal', coor)
    mid = gbuff.fetchpixelinfo('mid', coor)
    gui.text(f'position: [{pos.x:+.2f} {pos.y:+.2f} {pos.z:+.2f}]; texcoord: [{texcoor.x:.2f} {texcoor.y:.2f}]; normal: [{normal.x:+.2f} {normal.y:+.2f} {normal.z:+.2f}]; mid: {mid}', (0, 1))
    gui.show()

