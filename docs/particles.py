import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.Scene()

pars = tina.SimpleParticles()
scene.add_object(pars)

gui = ti.GUI('particles')

pos = np.random.rand(1024, 3).astype(np.float32) * 2 - 1
pars.set_particle_positions(pos)

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
