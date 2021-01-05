import taichi as ti
import tina

ti.init(ti.cpu)

scene = tina.Scene(smoothing=True)#, ibl=True)

#metallic = tina.Param(float)
#roughness = tina.Param(float)
#material = tina.CookTorrance(metallic=metallic, roughness=roughness)
material = tina.Lambert()
model = tina.PrimitiveMesh.sphere()
scene.add_object(model, material)

gui = ti.GUI('matball')
#metallic.make_slider(gui, 'metallic')
#roughness.make_slider(gui, 'roughness')

scene.init_control(gui)#, blendish=True)
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
