import taichi as ti
import tina

ti.init(ti.cpu)

scene = tina.Scene(smoothing=True, texturing=True)
scene.load_gltf('assets/cornell.gltf')
#scene.load_gltf('/home/bate/Downloads/军用飞机/DamagedHelmet.gltf')

gui = ti.GUI('gltf')
scene.init_control(gui, center=(0, 2, 0), radius=6)

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
