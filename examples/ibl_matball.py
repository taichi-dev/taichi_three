import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene(smoothing=True, taa=True, ibl=True)

roughness = tina.Param(float, initial=0)#.15)
metallic = tina.Param(float, initial=0)#.25)
material = tina.PBR(metallic=metallic, roughness=roughness)

#model = tina.PrimitiveMesh.sphere()
model = tina.MeshModel('assets/bunny.obj')
scene.add_object(model, material)

gui = ti.GUI('matball')
roughness.make_slider(gui, 'roughness')
metallic.make_slider(gui, 'metallic')

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
