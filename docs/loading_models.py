import taichi_three as t3

obj = t3.readobj('assets/cube.obj')

scene = t3.Scene()
camera = t3.Camera()
scene.add_camera(camera)

model = t3.SimpleModel(faces_n=len(obj['f']), pos_n=len(obj['vp']))
scene.add_model(model)

model.pos.from_numpy(obj['vp'] * 0.6)
model.clr.from_numpy(obj['vp'] * 0.5 + 0.5)
model.faces.from_numpy(obj['f'][:, :, 0])

gui = t3.GUI('Hello Triangle')
while gui.running:
    scene.render()
    gui.get_event(None)
    camera.from_mouse(gui)
    gui.set_image(camera.img)
    gui.show()