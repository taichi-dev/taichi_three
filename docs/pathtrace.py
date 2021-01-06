import taichi as ti
import numpy as np
import taichi_inject
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True, texturing=True)
material = tina.Phong()
mesh = tina.MeshModel('assets/monkey.obj')
scene.add_object(mesh, material)
pars = tina.SimpleParticles()
scene.add_object(pars, material)
pars.set_particles(np.random.rand(2**10, 3) * 2 - 1)

gui = ti.GUI('pathtrace', scene.res)

scene.update()
while gui.running:
    scene.input(gui)
    scene.render(nsteps=8)
    gui.set_image(scene.img)
    gui.show()
