import taichi as ti
import tina

ti.init(ti.opengl)

scene = tina.Scene(smoothing=True, ssr=True)
scene.load_gltf('/home/bate/Documents/testssr.gltf')

gui = ti.GUI(res=scene.res)

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
