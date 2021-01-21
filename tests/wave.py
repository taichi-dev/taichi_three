import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True)
#scene.lighting.skybox = tina.Atomsphere()
#scene.lighting.skybox = tina.Skybox('assets/grass.jpg')

scene.add_object(tina.MeshModel('assets/sphere.obj'), tina.Lambert())
scene.add_object(tina.MeshTransform(tina.MeshModel('assets/cube.obj'),
        tina.translate([0, -3, 0]) @ tina.scale(2)), tina.Lambert())

scene.lighting.set_lights(np.array([
    [0, 6, 0],
]))
scene.lighting.set_light_colors(np.array([
    [64, 64, 64],
]))
scene.lighting.set_light_radii(np.array([
    0.5,
]))

gui = ti.GUI('wave', scene.res)

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render(nsteps=8)
    gui.set_image(scene.img)
    gui.show()

tina.pfmwrite('/tmp/a.pfm', scene.img)
