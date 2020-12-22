import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene(smoothing=True)

model = tina.MeshSmoothNormal(tina.MeshModel('assets/monkey.obj'))
scene.add_object(model)

gui = ti.GUI('smooth')

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
