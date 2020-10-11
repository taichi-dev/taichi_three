import taichi_three as t3

scene = t3.Scene()
camera = t3.Camera()
scene.add_camera(camera)

model = t3.SimpleModel()
scene.add_model(model)

model.gl.Begin('GL_TRIANGLES')
model.gl.Color(1.0, 0.0, 0.0)
model.gl.Vertex(+0.0, +0.5, 0.0)
model.gl.Color(0.0, 1.0, 0.0)
model.gl.Vertex(+0.5, -0.5, 0.0)
model.gl.Color(0.0, 0.0, 1.0)
model.gl.Vertex(-0.5, -0.5, 0.0)
model.gl.End()

gui = t3.GUI('Hello Triangle')
while gui.running:
    scene.render()
    gui.get_event(None)
    camera.from_mouse(gui)
    gui.set_image(camera.img)
    gui.show()