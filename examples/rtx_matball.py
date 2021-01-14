import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True)
scene.lighting.skybox = tina.Atomsphere()

roughness = tina.Param(float, initial=0.15)
metallic = tina.Param(float, initial=1.0)
material = tina.PBR(metallic=metallic, roughness=roughness)

model = tina.PrimitiveMesh.sphere()
scene.add_object(model, material)

gui = ti.GUI('matball')
roughness.make_slider(gui, 'roughness')
metallic.make_slider(gui, 'metallic')

scene.update()
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
