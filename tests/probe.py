import tina
import taichi as ti


scene = tina.Scene(texturing=True, prob=True)

verts, faces = tina.readobj('assets/monkey.obj', simple=True)
mesh = tina.SimpleMesh()
texture = tina.LerpTexture(x0=[1.0, 1.0, 1.0], x1=[0.0, 0.0, 1.0])
material = tina.Diffuse(color=texture)
scene.add_object(mesh, material)

probe = tina.ProbeShader(scene.res)
scene.post_shaders.append(probe)

mesh.set_face_verts(verts[faces])

@ti.func
def ontouch(probe, I, r):
    e = probe.elmid[I]
    for i in ti.static(range(3)):
        mesh.coors[e, i].x = 1.0

gui = ti.GUI()
while gui.running:
    scene.input(gui)
    if gui.is_pressed(gui.LMB):
        mx, my = gui.get_cursor_pos()
        probe.touch(ontouch, mx, my, 0)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
