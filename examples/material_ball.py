import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
obj = t3.readobj('assets/sphere.obj', scale=0.9)
model = t3.Model.from_obj(obj)
model.add_texture('specular', np.array([[1.0]]))
model.add_texture('roughness', np.array([[0.5]]))
model.add_texture('metallic', np.array([[0.0]]))
scene.add_model(model)
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[0, 1, -1.8])
scene.add_camera(camera)
light = t3.Light(dir=[0.9, -1.5, 1.3], color=[0.78, 0.78, 0.78])
scene.add_light(light)
ambient = t3.AmbientLight(color=[0.24, 0.24, 0.24])
scene.add_light(ambient)

gui = ti.GUI('Material Ball', camera.res)
roughness = gui.slider('roughness', 0, 1, step=0.05)
metallic = gui.slider('metallic', 0, 1, step=0.05)
roughness.value = 0.3
metallic.value = 0.0
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    model.textures['roughness'].fill(roughness.value)
    model.textures['metallic'].fill(metallic.value)
    if any(x < 0.6 for x in gui.get_cursor_pos()):
        camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
