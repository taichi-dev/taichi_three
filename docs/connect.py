import taichi as ti
import tina

ti.init(ti.gpu)

verts, faces = tina.readobj('assets/monkey.obj', simple=True)

scene = tina.Scene()
mesh = tina.ConnectiveMesh()
scene.add_object(mesh, tina.Classic())

mesh.set_vertices(verts)
mesh.set_faces(faces)


gui = ti.GUI()
while gui.running:
    scene.input(gui)
    verts[100, 1] = ti.sin(gui.frame * 0.02)
    mesh.set_vertices(verts)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
