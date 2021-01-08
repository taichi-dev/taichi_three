import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.Scene()

pars = tina.SimpleParticles()
material = tina.Classic()
scene.add_object(pars, material)

gui = ti.GUI('particles')

pos = np.random.rand(1024, 3).astype(np.float32) * 2 - 1
pars.set_particles(pos)
radius = np.random.rand(1024).astype(np.float32) * 0.1 + 0.1
pars.set_particle_radii(radius)
color = np.random.rand(1024, 3).astype(np.float32) * 0.8 + 0.2
pars.set_particle_colors(color)

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
