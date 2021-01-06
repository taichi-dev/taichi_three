# In this episode, you'll learn how to use lights and materials in Tina
#
# This tutorial is based on docs/monkey.py, make sure you check that first

import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene()

# 5. Material - for describing the material of an object
material = tina.PBR(metallic=0.6, roughness=0.2)
# parameters may also be specified by textures (add texturing=True to Scene)
#material = tina.PBR(basecolor=tina.Texture('assets/cloth.jpg'))

model = tina.MeshModel('assets/monkey.obj')
# load our model into the scene with material specified:
scene.add_object(model, material)

gui = ti.GUI('lighting')

# now, let's add some custom light sources into the scene for test
#
# first of all, remove the 'default light' from scene:
scene.lighting.clear_lights()
# adds a directional light with direction (0, 0, 1), with white color
# the direction will be automatically normalized to obtain desired result
scene.lighting.add_light(dir=[0, 0, 1], color=[1, 1, 1])
# adds a point light at position (1, 1.5, 0.3), with red color
scene.lighting.add_light(pos=[1, 1.5, 0.3], color=[1, 0, 0])
# specifies the ambient color to be dark green
scene.lighting.set_ambient_light([0, 0.06, 0])

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
