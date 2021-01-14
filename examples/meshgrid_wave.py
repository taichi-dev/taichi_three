import taichi as ti
import time
import tina

ti.init(ti.gpu)


@ti.func
def Z(xy, t):
    return 0.1 * ti.sin(10 * xy.norm() - ti.tau * t)


@ti.kernel
def deform_mesh(t: float):
    for i, j in mesh.pos:
        mesh.pos[i, j].z = Z(mesh.pos[i, j].xy, t)


scene = tina.Scene(smoothing=True)

mesh = tina.MeshNoCulling(tina.MeshGrid(64))
scene.add_object(mesh)

gui = ti.GUI('meshgrid_wave', scene.res)

while gui.running:
    scene.input(gui)

    deform_mesh(gui.frame * 0.01)

    scene.render()
    gui.set_image(scene.img)
    gui.show()
