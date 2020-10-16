import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
obj = t3.readobj('assets/sphere.obj', scale=0.9)
model = t3.Model(t3.Mesh.from_obj(obj))
material = t3.Material(t3.CookTorrance(
    roughness=t3.Uniform((), float),
    metallic=t3.Uniform((), float),
    ))
scene.set_material(1, material)
scene.add_model(model)
camera = t3.Camera()
camera.fb.post_process = t3.make_tonemap()
camera.ctl = t3.CameraCtl(pos=[0, 1, 1.8])
scene.add_camera(camera)
light = t3.Light(dir=[0.9, -1.5, -1.3])
scene.add_light(light)

gui = ti.GUI('Material Ball', camera.res)
roughness = gui.slider('roughness', 0, 1, step=0.05)
metallic = gui.slider('metallic', 0, 1, step=0.05)
roughness.value = 0.3
metallic.value = 0.0
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    material.shader.params['roughness'].fill(roughness.value)
    material.shader.params['metallic'].fill(metallic.value)
    if any(x < 0.6 for x in gui.get_cursor_pos()):
        camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
