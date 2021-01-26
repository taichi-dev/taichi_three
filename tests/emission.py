import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.PTScene()
#scene.engine.skybox = tina.Atomsphere()

model = tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
        tina.translate([0, 0, 4]) @ tina.eularXYZ([ti.pi / 2, 0, 0]))
material = tina.Emission() * 2
scene.add_object(model, material)

metallic = tina.Param(float, initial=1.0)
roughness = tina.Param(float, initial=0.01)
model = tina.MeshModel('assets/monkey.obj')
material = tina.PBR(metallic=metallic, roughness=roughness)
scene.add_object(model, material)

gui = ti.GUI(res=scene.res)
metallic.make_slider(gui, 'metallic')
roughness.make_slider(gui, 'roughness')

scene.update()
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
