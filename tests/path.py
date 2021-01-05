import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene(smoothing=True, texturing=True)

#material = tina.Phong(color=[0.25, 0.5, 0.5])
material = tina.Glass()
scene.add_object(tina.PrimitiveMesh.sphere(), tina.Glass())
#scene.add_object(tina.MeshModel('assets/lens.obj'), tina.Glass())
#scene.add_object(tina.MeshTransform(tina.MeshModel('assets/cube.obj'), tina.translate([0, -1.2, 0]) @ tina.scale([2, 0.05, 2])), tina.Lambert())

gui = ti.GUI('path', scene.res)

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render(nsteps=5)
    gui.set_image(scene.img)
    gui.show()
