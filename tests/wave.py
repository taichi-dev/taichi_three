import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True)
#scene.lighting.skybox = tina.Atomsphere()
#scene.lighting.skybox = tina.Skybox('assets/grass.jpg')

scene.add_object(tina.PrimitiveMesh.sphere(), tina.Glass())
scene.add_object(tina.MeshTransform(tina.MeshModel('assets/cube.obj'),
        tina.translate([0, -3, 0]) @ tina.scale(2)), tina.Lambert())

scene.lighting.set_lights(np.array([
    [0, 5, 0],
]))
scene.lighting.set_light_colors(np.array([
    [16, 16, 16],
]))
scene.lighting.set_light_radii(np.array([
    0.01,
]))

gui = ti.GUI('wave', scene.res)

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render_light(nsteps=6)
    scene.render(nsteps=6)
    gui.set_image(scene.img)
    gui.show()

#tina.pfmwrite('/tmp/a.pfm', scene.img)
