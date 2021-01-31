import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True, texturing=True)

scene.add_object(tina.PrimitiveMesh.sphere(), tina.Glass())
scene.add_object(tina.MeshTransform(tina.MeshModel('assets/cube.obj'),
    tina.translate([0, -3, 0]) @ tina.scale(2)), tina.Lambert())

scene.add_object(tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
    tina.translate([0, 4, 0]) @ tina.scale(0.2)), tina.Lamp(color=tina.Texture("assets/cloth.jpg")) * 64)

gui = ti.GUI('bdpt', scene.res)

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render(nsteps=6)
    scene.render_light(nsteps=6)
    gui.set_image(scene.img)
    gui.show()
