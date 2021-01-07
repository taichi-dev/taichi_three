import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene(smoothing=True, taa=True, ibl=True)
model = tina.PrimitiveMesh.sphere()
roughness = tina.Param(float)
material = tina.CookTorrance(roughness=roughness)
#material = tina.Lambert()
scene.add_object(model, material)

gui = ti.GUI()
roughness.make_slider(gui, 'roughness')

scene.init_control(gui, blendish=True)
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()

