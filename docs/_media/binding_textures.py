import taichi_three as t3

scene = t3.Scene()
camera = t3.Camera()
scene.add_camera(camera)

light = t3.Light(dir=[-0.2, -0.6, 1.0])
scene.add_light(light)

obj = t3.readobj('cube.obj', scale=0.8)
model = t3.Model.from_obj(obj)
model.shading = t3.BlinnPhong
model.add_texture('color', t3.imread('container2.png'))
model.add_uniform('specular', 1.0)
scene.add_model(model)

gui = t3.GUI('Binding Textures')
while gui.running:
    scene.render()
    gui.get_event(None)
    camera.from_mouse(gui)
    gui.set_image(camera.img)
    gui.show()