import taichi_three as t3
import taichi as ti


scene = t3.Scene()
mesh = t3.DynamicMesh(n_faces=1, n_pos=3)
model = t3.Model(t3.MeshNoCulling(mesh))
scene.add_model(model)
camera = t3.Camera()
scene.add_camera(camera)
light = t3.AmbientLight(1.0)
scene.add_light(light)


@ti.kernel
def init_mesh():
    mesh.n_faces[None] = 1
    mesh.pos[0] = [ 0,  1,  0]
    mesh.pos[1] = [ 1, -1,  0]
    mesh.pos[2] = [-1, -1,  0]
    mesh.faces[0] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]


init_mesh()

gui = ti.GUI('Triangle', camera.res)
while gui.running:
    gui.get_event(None)
    gui.running = not gui.is_pressed(gui.ESCAPE)
    camera.from_mouse(gui)

    scene.render()
    gui.set_image(camera.img)
    gui.show()

