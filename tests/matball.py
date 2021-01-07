import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene(smoothing=True)#, taa=True, rtx=True)

metallic = tina.Param(float, initial=0.0)
specular = tina.Param(float, initial=0.5)
roughness = tina.Param(float, initial=0.4)
material = tina.PBR(metallic=metallic, roughness=roughness, specular=specular)

#shineness = tina.Param(float, initial=32)
#specular = tina.Param(float, initial=0.5)
#material = tina.Classic(shineness=shineness, specular=specular)

model = tina.PrimitiveMesh.sphere()
scene.add_object(model, material)

gui = ti.GUI('matball')
if 'roughness' in globals():
    roughness.make_slider(gui, 'roughness')
if 'metallic' in globals():
    metallic.make_slider(gui, 'metallic')
if 'shineness' in globals():
    shineness.make_slider(gui, 'shineness', 1, 500, 1)
if 'specular' in globals():
    specular.make_slider(gui, 'specular')

scene.init_control(gui, blendish=True)
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
