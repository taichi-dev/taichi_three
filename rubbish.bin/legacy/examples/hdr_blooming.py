import taichi as ti
import taichi_three as t3
import numpy as np

ti.init(ti.cpu)

scene = t3.Scene()
obj = t3.Geometry.cube()
model = t3.Model(t3.Mesh.from_obj(obj))
model.material = t3.Material(t3.BlinnPhong(
    # https://learnopengl.com/img/textures/container2.png
    color=t3.Texture('docs/_media/container2.png'),
    specular=t3.Texture('docs/_media/container2_specular.png'),
))
scene.add_model(model)
camera = t3.Camera()
scene.add_camera_d(camera)
original = t3.FrameBuffer(camera)
filted = t3.ImgUnaryOp(original, lambda x: ti.max(0, x - 1))
blurred = t3.GaussianBlur(filted, radius=21)
final = t3.ImgBinaryOp(original, blurred, lambda x, y: x + y)
scene.add_buffer(final)
light = t3.Light(dir=[-0.2, -0.6, -1.0], color=1.6)
scene.add_light(light)

gui_original = ti.GUI('Original', filted.res)
gui_filted = ti.GUI('Filted', filted.res)
gui_blurred = ti.GUI('Blurred', blurred.res)
gui_final = ti.GUI('Final', final.res)
gui_original.fps_limit = None
gui_filted.fps_limit = None
gui_blurred.fps_limit = None
gui_final.fps_limit = None
while gui_final.running and gui_filted.running and gui_blurred.running and gui_original.running:
    gui_final.get_event(None)
    camera.from_mouse(gui_final)
    scene.render()
    gui_original.set_image(original.img)
    gui_filted.set_image(filted.img)
    gui_blurred.set_image(blurred.img)
    gui_final.set_image(final.img)
    gui_original.show()
    gui_filted.show()
    gui_blurred.show()
    gui_final.show()
