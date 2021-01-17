import taichi as ti
import tina

ti.init(ti.opengl)


scene = tina.Scene(fxaa=True)
model = tina.MeshModel('assets/monkey.obj')
scene.add_object(model)#, tina.PBR(metallic=0.7, roughness=0.3))

gui = ti.GUI(res=scene.res)
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
