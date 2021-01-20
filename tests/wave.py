import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True)
scene.lighting.skybox = tina.Atomsphere()#Skybox('assets/grass.jpg')

scene.add_object(tina.MeshModel('assets/sphere.obj'), tina.CookTorrance(roughness=0.04))

gui = ti.GUI('wave', scene.res)

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render(nsteps=6)
    gui.set_image(scene.img)
    gui.show()
