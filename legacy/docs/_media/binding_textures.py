import taichi_three as t3

scene = t3.Scene()
camera = t3.Camera()
scene.add_camera(camera)

light = t3.Light(dir=[-0.2, -0.6, -1.0])
scene.add_light(light)

obj = t3.Geometry.cube()
model = t3.Model(t3.Mesh.from_obj(obj))
model.material = t3.Material(t3.BlinnPhong(
    color=t3.Texture('container2.png'),
    specular=t3.Texture('container2_specular.png'),
))
scene.add_model(model)

gui = t3.GUI('Binding Textures')
while gui.running:
    scene.render()
    gui.get_event(None)
    camera.from_mouse(gui)
    gui.set_image(camera.img)
    gui.show()