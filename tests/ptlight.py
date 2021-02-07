import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True)

roughness = tina.Param(float, initial=0.2)
metallic = tina.Param(float, initial=1.0)
scene.add_object(tina.MeshModel('assets/sphere.obj'), tina.PBR(metallic=metallic, roughness=roughness))

pars = tina.SimpleParticles()
scene.add_object(pars, tina.Lamp(color=32))

gui = ti.GUI('path', scene.res)
roughness.make_slider(gui, 'roughness', 0, 1, 0.01)
metallic.make_slider(gui, 'metallic', 0, 1, 0.01)

pars.set_particles(np.array([
    [0, 0, 5],
    ]))

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render(nsteps=6)
    gui.set_image(scene.img)
    gui.show()

#ti.imwrite(scene.img, 'output.png')
