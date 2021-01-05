import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene(smoothing=True, taa=True, rtx=True, ibl=False)

#roughness = tina.Param(float)
#material = tina.CookTorrance(roughness=roughness)

shineness = tina.Param(float, initial=32)
material = tina.Phong(shineness=shineness)

model = tina.PrimitiveMesh.sphere()
scene.add_object(model, material)

gui = ti.GUI('matball')
if 'roughness' in globals():
    roughness.make_slider(gui, 'roughness')
if 'metallic' in globals():
    metallic.make_slider(gui, 'metallic')
if 'shineness' in globals():
    shineness.make_slider(gui, 'shineness', 1, 400, 1)

scene.init_control(gui, blendish=True)
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
