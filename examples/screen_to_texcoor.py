import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
obj = t3.readobj('assets/torus.obj', scale=0.8)
model = t3.Model.from_obj(obj)
scene.add_model(model)
camera = t3.Camera(pos=[0, 1, -1.8], res=(600, 400))
camera.fb.add_buffer('texcoor', 2)
camera.fb.add_buffer('normal', 3)
scene.add_camera(camera)
light = t3.Light([0.4, -1.5, 1.8])
scene.add_light(light)

gui = ti.GUI('Model', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    pos = gui.get_cursor_pos()
    texcoor = camera.fb.fetchpixelinfo('texcoor', pos)
    normal = camera.fb.fetchpixelinfo('normal', pos)
    gui.text(f'texcoor: {texcoor}; normal: {normal}', (0, 0))
    gui.set_image(camera.img)
    gui.show()

