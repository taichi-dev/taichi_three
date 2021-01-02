import tina

scene = tina.Scene(smoothing=True, texturing=True)
scene.load_gltf('assets/cornell.gltf')

gui = tina.ti.GUI('gltf')
scene.init_control(gui, center=(0, 2, 0), radius=6)

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
