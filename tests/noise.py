import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True, texturing=True)
#scene.lighting.skybox = tina.Skybox('assets/skybox.jpg', cubic=True)
model = tina.MeshModel('assets/bunny.obj')
#material = tina.PBR(roughness=0.0, metallic=0.0)
material = tina.PBR(roughness=0.2, metallic=0.8)
scene.add_object(model, material)
denoise = tina.Denoise(scene.res)

if isinstance(scene, tina.PTScene):
    scene.update()

gui = ti.GUI('noise', scene.res)

while gui.running:
    scene.input(gui)
    if isinstance(scene, tina.PTScene):
        scene.render(nsteps=5)
    else:
        scene.render()
    #gui.set_image(scene.img)
    denoise.src.from_numpy(scene.img)
    denoise.nlm(radius=2, noiseness=0.9)
    gui.set_image(denoise.dst)
    gui.show()
