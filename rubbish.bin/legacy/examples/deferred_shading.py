import taichi as ti
import taichi_three as t3

ti.init(ti.cpu)

mtllib = [None]
scene = t3.Scene()
model = t3.Model(t3.Mesh.from_obj('assets/monkey.obj'))
model.material = t3.DeferredMaterial(mtllib, t3.Material(t3.CookTorrance()))
scene.add_model(model)
camera = t3.Camera()
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


gui = ti.GUI('Deferred Shading', imgbuf.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(imgbuf.img)
    gui.show()

