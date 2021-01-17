import taichi as ti
import tina

ti.init(ti.opengl)


scene = tina.Scene(fxaa=True)
model = tina.MeshModel('assets/monkey.obj')
scene.add_object(model, tina.PBR(metallic=0.5, roughness=0.3))


gui = ti.GUI(res=scene.res)

abs_thresh = gui.slider('abs_thresh', 0, 0.1, 0.002)
rel_thresh = gui.slider('rel_thresh', 0, 0.5, 0.01)
factor = gui.slider('factor', 0, 1, 0.01)
abs_thresh.value = 0.0625
rel_thresh.value = 0.063
factor.value = 1

while gui.running:
    scene.fxaa.rel_thresh[None] = rel_thresh.value
    scene.fxaa.abs_thresh[None] = abs_thresh.value
    scene.fxaa.factor[None] = factor.value
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
