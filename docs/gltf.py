import tina
from tina.assimp.gltf import readgltf

scene = tina.Scene()
readgltf('assets/cornell.gltf').extract(scene)

gui = tina.ti.GUI('gltf')

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
