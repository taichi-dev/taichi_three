import taichi as ti
import numpy as np
import tina

ti.init(ti.gpu)

scene = tina.PTScene()
verts, faces = tina.readobj('assets/sphere.obj', simple=True)
verts = verts[faces]
model = tina.SimpleMesh()
scene.add_object(model)
scene.engine.skybox = tina.Atomsphere()
scene.materials = [tina.Lambert(), tina.Lambert() * [0, 1, 0]]

mtlids = np.zeros(verts.shape[0])
mtlids[3] = 1
mtlids[5] = 1
model.set_face_verts(verts)
model.set_face_mtlids(mtlids)

scene.update()
scene.visualize()
