import taichi as ti
import numpy as np
import tina

ti.init(ti.cpu)

scene = tina.Scene(smoothing=True)

material = tina.BlinnPhong()
pars = tina.SimpleParticles()
#std = tina.MeshModel('assets/sphere.obj', scale=0.25)
scene.add_object(pars, material)
#scene.add_object(std, material)

gui = ti.GUI('particles')

pos = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
pars.set_particles(pos)
radius = np.array([1.0], dtype=np.float32)
pars.set_particle_radii(radius)
color = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
pars.set_particle_colors(color)

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
