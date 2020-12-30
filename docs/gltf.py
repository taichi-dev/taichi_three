import tina

scene = tina.Scene()
scene.load_gltf('assets/cornell.gltf')

gui = tina.ti.GUI('gltf')

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
