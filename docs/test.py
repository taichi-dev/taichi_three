import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

dens = np.load('assets/smoke.npy') * 32
scene = tina.Scene((1024, 768), N=dens.shape[0], taa=True)
volume = tina.SimpleVolume(dens.shape[0])
scene.add_object(volume)

gui = ti.GUI('test', scene.res, fast_gui=True)

volume.dens.from_numpy(dens)
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
