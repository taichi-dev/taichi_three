import taichi_three as t3

scene = t3.Scene()
camera = t3.Camera()
scene.add_camera(camera)

light = t3.Light(dir=[-0.2, -0.6, 1.0])
scene.add_light(light)
ambient = t3.AmbientLight(0.1)
scene.add_light(ambient)

obj = t3.Geometry.cylinder()
model = t3.Model.from_obj(obj)
scene.add_model(model)

gui = t3.GUI('Hello Cube')
while gui.running:
    gui.get_event(None)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()