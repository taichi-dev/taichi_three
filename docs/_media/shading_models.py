import taichi_three as t3

scene = t3.Scene()
camera = t3.Camera()
scene.add_camera(camera)

light = t3.Light(dir=[0.3, -1.0, 0.8])
scene.add_light(light)

obj = t3.readobj('assets/cube.obj', scale=0.8)
model = t3.Model.from_obj(obj)
scene.add_model(model)

gui = t3.GUI('Shading Models')
while gui.running:
    scene.render()
    gui.get_event(None)
    camera.from_mouse(gui)
    gui.set_image(camera.img)
    gui.show()