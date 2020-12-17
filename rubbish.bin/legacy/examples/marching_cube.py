import taichi as ti
import taichi_three as t3
from taichi_three.mciso import MCISO, Voxelizer
import numpy as np

ti.init(arch=ti.opengl)


vol = np.load('assets/smoke.npy')

mciso = MCISO(vol.shape[0], use_sparse=False)


scene = t3.Scene()
mesh = t3.DynamicMesh(n_faces=mciso.N_res, n_pos=mciso.N_res, n_nrm=mciso.N_res)
model = t3.Model(mesh)
scene.add_model(model)
camera = t3.Camera()
scene.add_camera(camera)
scene.add_light(t3.Light([0.4, -1.5, -1.8], 0.8))
scene.add_light(t3.AmbientLight(0.22))

@ti.kernel
def update_mesh():
    mesh.n_faces[None] = mciso.Js_n[None]
    for i in range(mciso.Js_n[None]):
        for t in ti.static(range(3)):
            mesh.faces[i][t, 0] = mciso.Jts[i][t]
            mesh.faces[i][t, 2] = mciso.Jts[i][t]
        mesh.pos[i] = (mciso.vs[i] + 0.5) / mciso.N * 2 - 1
        mesh.nrm[i] = mciso.ns[i]

mciso.clear()
mciso.m.from_numpy(vol * 4)
print(mciso.m.to_numpy().max())
mciso.march()
update_mesh()

gui = ti.GUI('MCISO', camera.res)
while gui.running:
    gui.get_event(None)
    camera.from_mouse(gui)
    if gui.is_pressed(gui.ESCAPE):
        gui.running = False

    scene.render()
    gui.set_image(camera.img)
    gui.show()
