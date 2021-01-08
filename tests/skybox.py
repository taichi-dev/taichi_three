import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.Scene(ibl=True, smoothing=True, taa=True, maxfaces=2**18)
#scene = tina.Scene(taa=True, rtx=True)
scene.lighting.load_skybox(tina.Atomsphere())

metallic = tina.Param(float, initial=0.0)
roughness = tina.Param(float, initial=0.0)
material = tina.PBR(metallic=metallic, roughness=roughness)
scene.add_object(tina.MeshModel('assets/bunny.obj'), material)

gui = ti.GUI('sky', scene.res)
metallic.make_slider(gui, 'metallic')
roughness.make_slider(gui, 'roughness')

scene.init_control(gui, blendish=True)
while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
