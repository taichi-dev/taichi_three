import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene(smoothing=True, taa=True, rtx=True)
model = tina.PrimitiveMesh.sphere()
material = tina.CookTorrance(roughness=0.2)
scene.add_object(model, material)

gui = ti.GUI()
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()

