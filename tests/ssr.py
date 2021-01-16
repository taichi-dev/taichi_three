import taichi as ti
import tina

ti.init(ti.opengl)

scene = tina.Scene(smoothing=True, ssr=True)
scene.add_object(tina.MeshModel('assets/monkey.obj'), tina.Diffuse())
scene.add_object(tina.MeshTransform(tina.MeshGrid(32),
    tina.scale(2) @ tina.eularXYZ([-ti.pi / 2, 0, 0])),
    tina.PBR(metallic=1.0, roughness=0.1))

gui = ti.GUI(res=scene.res)

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
