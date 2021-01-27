import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

dens = np.load('assets/smoke.npy')[::4, ::4, ::4]
scene = tina.PTScene()
scene.engine.skybox = tina.Atomsphere()
volume = tina.SimpleVolume(N=dens.shape[0])
scene.add_object(tina.MeshModel('assets/monkey.obj'))
scene.add_object(volume, tina.VolScatter() * [0.8, 0.9, 0.8])

gui = ti.GUI('volume', scene.res)

volume.set_volume_density(dens)
scene.update()
while gui.running:
    scene.input(gui)
    scene.render()#nsteps=32)
    gui.set_image(scene.img)
    gui.show()
