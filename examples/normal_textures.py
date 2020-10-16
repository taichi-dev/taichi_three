import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
obj = t3.readobj('assets/cube.obj', scale=0.6)
model = t3.Model(t3.Mesh.from_obj(obj))
model.material = t3.Material(t3.CookTorrance(
    color=t3.Texture(ti.imread('assets/cloth.jpg')),
    normal=t3.NormalMap(texture=t3.Texture(ti.imread('assets/normal.png'))),
))
scene.add_model(model)
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[0, 1, -1.8])
scene.add_camera(camera)
light = t3.Light([0.4, -1.5, 1.8])
scene.add_light(light)

gui = ti.GUI('Normal map', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    #gui.set_image(camera.fb['normal'].to_numpy() * 0.5 + 0.5)
    gui.show()
