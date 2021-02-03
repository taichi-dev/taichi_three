import taichi as ti
import numpy as np
import tina

ti.init(ti.opengl)

scene = tina.PTScene(smoothing=True)

roughness = tina.Param(float, initial=0.2)
metallic = tina.Param(float, initial=1.0)
scene.add_object(tina.MeshModel('assets/sphere.obj'), tina.PBR(metallic=metallic, roughness=roughness))

scene.add_object(tina.MeshTransform(tina.MeshModel('assets/plane.obj'),
        tina.translate([0, 0, 4]) @ tina.eularXYZ([ti.pi / 2, 0, 0])
        #@ tina.scale(0.1)), tina.Lamp(color=32))
        @ tina.scale(0.4)), tina.Lamp(color=1))

gui = ti.GUI('path', scene.res)
roughness.make_slider(gui, 'roughness', 0, 1, 0.01)
metallic.make_slider(gui, 'metallic', 0, 1, 0.01)

scene.update()
while gui.running:
    if scene.input(gui):
        scene.clear()
    scene.render(nsteps=6)
    gui.set_image(scene.img)
    gui.show()

#ti.imwrite(scene.img, 'output.png')
