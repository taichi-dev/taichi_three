import taichi_three as t3
import taichi as ti

N = 64

scene = t3.Scene()
mesh = t3.DynamicMesh(n_faces=N, n_pos=N + 1)
model = t3.Model(mesh)
scene.add_model(model)
camera = t3.Camera()
scene.add_camera(camera)
light = t3.AmbientLight(1.0)
scene.add_light(light)


@ti.materialize_callback
@ti.kernel
def init_mesh():
    mesh.nrm[0] = [0, 0, 1]
    mesh.pos[N] = [0, 0, 0]
    for i in range(N):
        a = i / N * t3.tau
        mesh.pos[i] = [ti.sin(a), ti.cos(a), 0]
        mesh.faces[i] = [[i, 0, 0], [(i + 1) % N, 0, 0], [N, 0, 0]]


gui = ti.GUI('Dynamic', camera.res)
while gui.running:
    mesh.n_faces[None] = gui.frame % N
    scene.render()
    gui.set_image(camera.img)
    gui.show()