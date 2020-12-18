# In this episode, you'll learn how to use lights and materials in Tina
#
# This tutorial is based on examples/monkey.py, make sure you check that first

import taichi as ti
import tina

ti.init(ti.gpu)

obj = tina.readobj('assets/monkey.obj')
verts = tina.objverts(obj)

engine = tina.Engine()

# 5. Lighting - for describing the lighting conditions
lighting = tina.Lighting()
# 6. Material - for describing the material of an object
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

# adds a directional light with direction (0, 0, 1), with white color
# the direction will be automatically normalized to obtain desired result
lighting.add_light(dir=[0, 0, 1], color=[1, 1, 1])
# adds a point light at position (1, 1.5, 0.3), with red color
lighting.add_light(pos=[1, 1.5, 0.3], color=[1, 0, 0])
# specifies the ambient color to be dark green
lighting.set_ambient_light([0, 0.06, 0])

while gui.running:
    control.get_camera(engine)

    img.fill(0)
    engine.clear_depth()

    engine.set_face_verts(verts)
    engine.render(shader)

    gui.set_image(img)
    gui.show()
