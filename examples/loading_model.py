import taichi as ti
import taichi_three as t3

ti.init(ti.cuda)
scene = t3.Scene((1024, 1024))

model = t3.Model(t3.readobj('assets/test.obj', 'assets/test.jpg', scale=0.55))
model.init()

scene.opt.diffuse = 0.4
scene.camera.set([0, 0.5, -1], [0, 1, 0], [0, 1, 0])
scene.add_model(model)
scene.set_light_dir([0.4, -1.5, -1.8])
gui = ti.GUI('Model', scene.res)
while gui.running:
    gui.running = not gui.get_event(ti.GUI.ESCAPE)
    model.L2W.from_mouse(gui)
    scene._render()
    gui.set_image(scene.img)
    gui.show()
