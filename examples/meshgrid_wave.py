import taichi as ti
import taichi_three as t3

ti.init(ti.cpu)

scene = t3.Scene()
mesh = t3.MeshGrid(2)
model = t3.WireframeModel(t3.PolyToEdge(t3.QuadToTri(mesh)))
scene.add_model(model)
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[1.1, 1.6, 1.6])
scene.add_camera(camera)
light = t3.Light([0.4, -1.5, -0.8], 0.9)
scene.add_light(light)
ambient = t3.AmbientLight(0.1)
scene.add_light(ambient)

@ti.func
def Z(xy, t):
    return 0.1 * ti.sin(10 * xy.norm() - t3.tau * t)

@ti.kernel
def deform_mesh(t: float):
    for i, j in mesh.pos:
        mesh.pos[i, j].y = Z(mesh.pos[i, j].xZ, t)


gui = ti.GUI('Meshgrid', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    #deform_mesh(t3.get_time())
    scene.render()
    gui.set_image(camera.img)
    gui.show()
