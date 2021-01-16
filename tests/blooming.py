import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene(smoothing=True, blooming=True)

material = tina.Diffuse()
model = tina.MeshModel('assets/monkey.obj')
scene.add_object(model, material)

gui = ti.GUI(res=scene.res)
light = gui.slider('light', 0, 32, 0.1)
light.value = 6

scene.lighting.clear_lights()

while gui.running:
    scene.lighting.set_ambient_light([light.value] * 3)
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
