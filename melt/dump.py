from tina.advans import *

pars = np.load('/tmp/mpm.npy')

scene = tina.Scene()
model = tina.SimpleParticles()
scene.add_object(model)

gui = ti.GUI()
while gui.running:
    scene.input(gui)
    model.set_particles(pars)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
