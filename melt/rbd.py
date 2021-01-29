from tina.advans import *

_, __ = tina.readobj('assets/monkey.obj', simple=True)

verts_np = _[__]
verts = texture_as_field(verts_np)

dt = 0.002
invM = 1 / 4
invI = 1 / 8
Ks = 8000
Kd = 20
pos = ti.Vector.field(3, float, ())
vel = ti.Vector.field(3, float, ())
rot = ti.Vector.field(3, float, ())
anv = ti.Vector.field(3, float, ())

scene = tina.Scene()
model = tina.MeshTransform(tina.SimpleMesh())
scene.add_object(model)
#passive = tina.MeshTransform(tina.SimpleMesh())
#scene.add_object(passive)

@ti.kernel
def reset():
    pos[None] = 0
    vel[None] = 0
    rot[None] = 0
    anv[None] = 0

@ti.kernel
def forward():
    pos[None] += vel[None] * dt
    rot[None] += anv[None] * dt
    vel[None].y -= 0.98

@ti.kernel
def collide():
    force = V(0., 0., 0.)
    torque = V(0., 0., 0.)
    for i, j in verts:
        p = mapply_pos(model.trans[None], verts[i, j])
        if p.y >= -1.5:
            continue
        F = Ks * (-1.5 - p.y) * U3(1)
        curvel = vel[None] + anv[None].cross(verts[i, j])
        F += -Kd * curvel
        torque += verts[i, j].cross(F)
        force += F
    vel[None] += force * dt * invM
    anv[None] += torque * dt * invI

def step():
    for i in range(2):
        forward()
        trans = tina.translate(pos[None].value) @ tina.eularXYZ(rot[None].value)
        model.set_transform(trans)
        collide()

reset()
model.set_face_verts(verts_np)
gui = ti.GUI('rbd')
while gui.running:
    scene.input(gui)
    if not gui.is_pressed(gui.SPACE):
        step()
    if gui.is_pressed('r'):
        reset()
    scene.render()
    gui.set_image(scene.img)
    gui.show()
