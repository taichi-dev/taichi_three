import taichi as ti
import tina

ti.init(ti.cpu)


scene = tina.Scene()
model = tina.MeshModel('assets/cube.obj')
material = tina.Emission() * tina.Input('pos')
scene.add_object(model, material)

gui = ti.GUI(res=scene.res)

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
