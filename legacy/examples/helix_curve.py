import taichi_three as t3
import taichi as ti

N = 512

scene = t3.Scene()
mesh = t3.DynamicMesh(n_faces=N - 1, n_pos=N)
model = t3.WireframeModel(t3.PolyToEdge(mesh))
scene.add_model(model)
camera = t3.Camera()
scene.add_camera(camera)


@ti.kernel
def init_mesh():
    for i in range(N):
        x = i / N * 2 - 1
        mesh.pos[i] = [x, ti.sin(x * 10), ti.cos(x * 10)]

    for i in range(N - 1):
        mesh.faces[i] = [[i, 0, 0], [i, 0, 0], [i + 1, 0, 0]]
    mesh.n_faces[None] = N - 1


init_mesh()

gui = ti.GUI('Helix', camera.res)
while gui.running and not gui.get_event(gui.ESCAPE):
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()
