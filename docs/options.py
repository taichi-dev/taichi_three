# In this episode, you'll learn some basic options to specify for a Tina scene.
#
# This tutorial is based on docs/monkey.py, make sure you check that first

import taichi as ti
import tina

ti.init(ti.gpu)

# There are some options you may specify to tina.Scene, try turn off some
# of them and see what's the difference
#
# culling: enable face culling for better performance (default: on)
# clipping: enable view space clipping for rid objects out of depth (default: off)
# smoothing: enable smooth shading by interpolating normals (default: off)
# texturing: enable texture coordinates, see docs/texture.py (default: off)
# taa: temporal anti-aliasing, won't work well for dynamic scenes (default: off)
scene = tina.Scene(smoothing=True, taa=True)

# (make sure your OBJ model have normals to make smooth shading work)
model = tina.MeshModel('assets/torus.obj')
scene.add_object(model)

gui = ti.GUI('options')

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
