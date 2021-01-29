from tina.advans import *

_, __ = tina.readobj('assets/monkey.obj', simple=True)
verts_np = _[__]
verts = texture_as_field(verts_np)

_, __ = tina.readobj('assets/sphere.obj', simple=True)
passes_np = _[__]
passes_np[:, :, 1] -= 2.5
passes = texture_as_field(passes_np)

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
passive = tina.MeshTransform(tina.SimpleMesh())
scene.add_object(passive)

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

@ti.func
def hit(curpos):
    ret_sdf = -inf
    ret_norm = V(0., 0., 0.)
    for p in range(passes.shape[0]):
        e1 = passes[p, 1] - passes[p, 0]
        e2 = passes[p, 2] - passes[p, 0]
        norm = e1.cross(e2).normalized()
        sdf = (curpos - passes[p, 0]).dot(norm)
        if sdf > ret_sdf:
            ret_sdf = sdf
            ret_norm = norm
    return ret_sdf, ret_norm

@ti.kernel
def collide():
    force = V(0., 0., 0.)
    torque = V(0., 0., 0.)
    for i, j in verts:
        curpos = mapply_pos(model.trans[None], verts[i, j])
        curvel = vel[None] + anv[None].cross(verts[i, j])
        sdf, norm = hit(curpos)
        if sdf >= 0:
            continue
        F = -Ks * sdf * norm - Kd * curvel
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
passive.set_face_verts(passes_np)
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
