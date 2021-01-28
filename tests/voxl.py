import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

dens = np.load('assets/smoke.npy')[::1, ::1, ::1]
scene = tina.PTScene()
scene.engine.skybox = tina.Atomsphere()
volume = tina.VolumeScale(tina.SimpleVolume(N=dens.shape[0]), scale=5)
scene.add_object(tina.MeshModel('assets/monkey.obj'))
g = tina.Param(float, initial=0.76)
scene.add_object(volume, tina.HenyeyGreenstein(g=g))
#scene.add_object(tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
#    tina.translate([0, 0, 4]) @ tina.eularXYZ([ti.pi / 2, 0, 0])),
#    tina.Emission() * 4)

gui = ti.GUI('volume', scene.res)
g.make_slider(gui, 'g', -1, 1, 0.01)

volume.set_volume_density(dens)
scene.update()
while gui.running:
    scene.input(gui)
    scene.render()#nsteps=32)
    gui.set_image(scene.img)
    gui.show()
