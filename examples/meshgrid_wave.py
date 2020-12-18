import taichi as ti
import time
import tina
from tina.geom.grid import MeshGrid

ti.init(ti.gpu)


mesh = MeshGrid(64)

@ti.func
def Z(xy, t):
    return 0.1 * ti.sin(10 * xy.norm() - ti.tau * t)

@ti.kernel
def deform_mesh(t: float):
    for i, j in mesh.pos:
        mesh.pos[i, j].z = Z(mesh.pos[i, j].xy, t)


engine = tina.Engine(smoothing=True, culling=False)

img = ti.Vector.field(3, float, engine.res)
shader = tina.SimpleShader(img)

gui = ti.GUI('meshgrid_wave', engine.res)
control = tina.Control(gui)

while gui.running:
    control.get_camera(engine)

    deform_mesh(time.time() % 1e5)

    img.fill(0)
    engine.clear_depth()

    engine.set_mesh(mesh)
    engine.render(shader)

    gui.set_image(img)
    gui.show()
