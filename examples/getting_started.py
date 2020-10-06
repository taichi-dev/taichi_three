import taichi_three as t3

scene = t3.Scene()
model = t3.ModelEZ(faces_n=1, pos_n=3)
scene.add_model(model)

camera = t3.Camera()
scene.add_camera(camera)

light = t3.Light([0.4, -1.5, -1.8])
scene.add_light(light)

model.pos[0] = [+0.0, +0.5, 0.0]
model.pos[1] = [+0.5, -0.5, 0.0]
model.pos[2] = [-0.5, -0.5, 0.0]
model.clr[0] = [1.0, 0.0, 0.0]
model.clr[1] = [0.0, 1.0, 0.0]
model.clr[2] = [0.0, 0.0, 1.0]
model.faces[0] = [0, 1, 2]

gui = t3.GUI('Hello Triangle')
while gui.running:
    scene.render()
    gui.get_event(None)
    camera.from_mouse(gui)
    gui.set_image(camera.img)
    gui.show()