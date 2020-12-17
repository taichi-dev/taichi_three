# In this episode, you'll learn how to use lights and materials in Tina
#
# This tutorial is based on examples/monkey.py, make sure you check that first

import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

obj = tina.readobj('assets/monkey.obj')
verts = tina.objverts(obj)

engine = tina.Engine()
camera = tina.Camera()

# 6. Lighting - for describing the lighting conditions
lighting = tina.Lighting()
# 7. Material - for describing the material of an object
material = tina.BlinnPhong()
# you may also specify some parameters for the Blinn-Phong material:
#material = tina.BlinnPhong(shineness=10, diffuse=[1, 0, 0])
# or use the Cook-Torrance material for PBR:
#material = tina.CookTorrance(metallic=0.6, roughness=0.2)

img = ti.Vector.field(3, float, engine.res)
# unlike the dummy tina.SimpleShader we used before, `tina.Shader` can consider
# real light conditions and materials therefore producing more realistic result
shader = tina.Shader(img, lighting, material)

gui = ti.GUI('lighting')
control = tina.Control(gui)

while gui.running:
    control.get_camera(camera)
    engine.set_camera(camera)

    # specify the number of lights being used:
    lighting.nlights[None] = 2

    # adds a directional light with direction (0, 0, 1), with white color
    # the direction should be normalized to get desired result
    lighting.light_dirs[0] = [0, 0, 1, 0]
    lighting.light_colors[0] = [1, 1, 1]
    # adds a point light at position (1, 1.5, 0.3), with red color
    lighting.light_dirs[1] = [1, 1.5, 0.3, 1]
    lighting.light_colors[1] = [1, 0, 0]

    # specifies the ambient color to be dark green
    lighting.ambient_color[None] = [0, 0.06, 0]

    img.fill(0)
    engine.clear_depth()

    engine.set_face_verts(verts)
    engine.render(shader)

    gui.set_image(img)
    gui.show()
