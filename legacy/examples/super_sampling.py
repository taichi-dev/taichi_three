import taichi as ti
import taichi_three as t3

ti.init(ti.cpu)

scene = t3.Scene()
model = t3.Model(t3.Mesh.from_obj('assets/monkey.obj'))
scene.add_model(model)
camera = t3.Camera(res=(1024, 1024))
scene.add_camera_d(camera)
buffer = t3.SuperSampling2x2(t3.FrameBuffer(camera))
scene.add_buffer(buffer)
light = t3.Light([0.4, -1.5, -0.8], 0.9)
scene.add_light(light)
ambient = t3.AmbientLight(0.1)
scene.add_light(ambient)


gui = ti.GUI('SSAA 2x2', buffer.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(buffer.img)
    gui.show()

