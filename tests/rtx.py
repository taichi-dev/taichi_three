import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene(smoothing=True, rtx=True)

material = tina.Phong(shineness=256)

model = tina.PrimitiveMesh.sphere()
scene.add_object(model, material)

moving = tina.MeshTransform(tina.PrimitiveMesh.sphere(), tina.translate([1.4, 0, 0]) @ tina.scale(0.5))
scene.add_object(moving, material)

gui = ti.GUI('matball')

scene.init_control(gui)
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
