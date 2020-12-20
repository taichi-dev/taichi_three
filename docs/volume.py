import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

dens = np.load('assets/smoke.npy')
scene = tina.Scene(N=dens.shape[0], taa=True, density=16)
volume = tina.SimpleVolume(N=dens.shape[0])
#model = tina.MeshModel('assets/monkey.obj')
#scene.add_object(model, tina.CookTorrance(metallic=0.8))
scene.add_object(volume)

gui = ti.GUI('volume', scene.res)

volume.set_volume_density(dens)
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
