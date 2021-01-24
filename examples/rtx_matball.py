import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True)
scene.engine.skybox = tina.Atomsphere()
#scene.engine.skybox = tina.Skybox('assets/skybox.jpg')

roughness = tina.Param(float, initial=0.15)
metallic = tina.Param(float, initial=1.0)
specular = tina.Param(float, initial=0.5)
material = tina.PBR(metallic=metallic, roughness=roughness, specular=specular)

model = tina.PrimitiveMesh.sphere()
scene.add_object(model, material)

gui = ti.GUI('matball')
roughness.make_slider(gui, 'roughness')
metallic.make_slider(gui, 'metallic')
specular.make_slider(gui, 'specular')

scene.update()
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()

#tina.pfmwrite('/tmp/color.pfm', scene.img)
