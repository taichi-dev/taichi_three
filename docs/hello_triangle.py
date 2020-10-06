import taichi_three as t3

scene = t3.Scene()
camera = t3.Camera()
scene.add_camera(camera)

model = t3.SimpleModel(faces_n=2, pos_n=3)
scene.add_model(model)

model.pos[0] = [+0.0, +0.5, 0.0]
model.pos[1] = [+0.5, -0.5, 0.0]
model.pos[2] = [-0.5, -0.5, 0.0]
model.clr[0] = [1.0, 0.0, 0.0]
model.clr[1] = [0.0, 1.0, 0.0]
model.clr[2] = [0.0, 0.0, 1.0]
model.faces[0] = [0, 1, 2]
model.faces[1] = [0, 2, 1]

gui = t3.GUI('Hello Triangle')
while gui.running:
    scene.render()
    gui.get_event(None)
    camera.from_mouse(gui)
    gui.set_image(camera.img)
    gui.show()