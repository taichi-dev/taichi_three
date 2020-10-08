import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
obj = t3.readobj('assets/torus.obj', scale=0.8)
model = t3.Model.from_obj(obj)
scene.add_model(model)
camera = t3.Camera(pos=[0, 1, -1.8], res=(600, 400))
camera.fb.add_buffer('pos', 3)
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
    gui.set_image(camera.img)
    coor = gui.get_cursor_pos()
    pos = camera.fb.fetchpixelinfo('pos', coor)
    color = camera.fb.fetchpixelinfo('img', coor)
    texcoor = camera.fb.fetchpixelinfo('texcoor', coor)
    normal = camera.fb.fetchpixelinfo('normal', coor)
    gui.text(f'color: [{color.x:.2f} {color.y:.2f} {color.z:.2f}]; pos: [{pos.x:+.2f} {pos.y:+.2f} {pos.z:+.2f}]; texcoor: [{texcoor.x:.2f} {texcoor.y:.2f}]; normal: [{normal.x:+.2f} {normal.y:+.2f} {normal.z:+.2f}]', (0, 1))
    gui.show()

