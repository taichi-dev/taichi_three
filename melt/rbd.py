from tina.advans import *

_, __ = tina.readobj('assets/cube.obj', simple=True)
verts_np = _[__]
verts = texture_as_field(verts_np)

_, __ = tina.readobj('assets/sphere.obj', simple=True)
passes_np = _[__]
passes_np[:, :, 1] -= 2.5
passes = texture_as_field(passes_np)

dt = 0.001
invM = 1 / 4
invI = 1 / 16
Ks = 16000
Kd = 256
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

@ti.func
def hitedge(cp1, cp2):
    ret_sdf = -inf
    ret_norm = V(0., 0., 0.)
    ret_coor = 0.
    for p in range(passes.shape[0]):
        e1 = passes[p, 1] - passes[p, 0]
        e2 = passes[p, 2] - passes[p, 0]
        norm = e1.cross(e2).normalized()
        sdf1 = (cp1 - passes[p, 0]).dot(norm)
        sdf2 = (cp2 - passes[p, 0]).dot(norm)
        sdf = min(sdf1, sdf2)
        if sdf > ret_sdf:
            ret_coor = ti.sqrt(clamp(
                (sdf1**2 + eps) / ((sdf1 - sdf2)**2 + 2 * eps), 0., 1.))
            ret_sdf = sdf
            ret_norm = norm
    return ret_sdf, ret_norm, ret_coor

@ti.func
def curposvel(relpos):
    curpos = mapply_pos(model.trans[None], relpos)
    curvel = vel[None] + anv[None].cross(relpos)
    return curpos, curvel

@ti.kernel
def collide():
    force = V(0., 0., 0.)
    torque = V(0., 0., 0.)
    '''
    for i, j in verts:
        curpos, curvel = curposvel(verts[i, j])
        sdf, norm = hit(curpos)
        if sdf >= 0:
            continue
        F = -Ks * sdf * norm - Kd * curvel
        torque += verts[i, j].cross(F)
        force += F
    '''
    for i in range(verts.shape[0]):
        for e in range(3):
            p1, p2 = verts[i, e], verts[i, (e + 1) % 3]
            cp1, cv1 = curposvel(p1)
            cp2, cv2 = curposvel(p2)
            sdf, norm, coor = hitedge(cp1, cp2)
            edgelen = (p1 - p2).norm()
            relpos = lerp(coor, p1, p2)
            curpos = lerp(coor, cp1, cp2)
            curvel = lerp(coor, cv1, cv2)
            if sdf >= 0:
                continue
            F = -Ks * sdf * norm - Kd * curvel
            F *= edgelen
            torque += relpos.cross(F)
            force += F
    vel[None] += force * dt * invM
    anv[None] += torque * dt * invI

def step():
    for i in range(3):
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
