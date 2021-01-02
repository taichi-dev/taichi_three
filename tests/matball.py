import taichi as ti
import tina

ti.init(ti.cpu)

scene = tina.Scene(smoothing=True)

metallic = tina.Param(float)
roughness = tina.Param(float)
material = tina.CookTorrance(metallic=metallic, roughness=roughness)
model = tina.MeshModel('assets/sphere.obj')
scene.add_object(model, material)

gui = ti.GUI('matball')
metallic.make_slider(gui, 'metallic')
roughness.make_slider(gui, 'roughness')

scene.init_control(gui, blendish=True)
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(tina.aces_tonemap(scene.img.to_numpy()))
    gui.show()
