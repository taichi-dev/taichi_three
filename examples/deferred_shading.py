import taichi as ti
import taichi_three as t3

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model(t3.Mesh.from_obj('assets/monkey.obj'))
model.material = t3.DeferredMaterial()
scene.add_model(model)
camera = t3.Camera()
scene.add_camera(camera)
buffer = t3.DeferredShading(t3.FrameBuffer(camera, dim=14), material=t3.Material(t3.CookTorrance()))
scene.add_buffer(buffer)
light = t3.Light([0.4, -1.5, -0.8], 0.9)
scene.add_light(light)
ambient = t3.AmbientLight(0.1)
scene.add_light(ambient)


gui = ti.GUI('Meshgrid', buffer.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(buffer.img)
    gui.show()

