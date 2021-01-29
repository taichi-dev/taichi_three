# https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-29-real-time-rigid-body-simulation-gpus

from tina.advans import *
#from voxelizer import MeshVoxelizer

NE = 64
N = 128
R = 0.1
Ks = 600
Kd = 6
Kt = 0
Dt = 0.001
Grav = 2.5
pos = ti.Vector.field(3, float, N)
vel = ti.Vector.field(3, float, N)
epos = ti.Vector.field(3, float, NE)
erot = ti.Vector.field(4, float, NE)

@ti.kernel
def reset():
    for i in pos:
        pos[i] = V(ti.random(), ti.random(), ti.random()) * 2 - 1
        vel[i] = 0

@ti.kernel
def substep():
    for i in pos:
        acc = V(0., 0., 0.)
        for j in range(N):
            if i == j:
                continue
            r = pos[j] - pos[i]
            v = vel[j] - vel[i]
            rn = r.norm()
            if rn > R * 2:
                continue
            rnn = r / rn
            fs = -Ks * (R * 2 - rn) * rnn
            vd = v.dot(rnn) * rnn
            ft = Kt * (v - vd)
            fd = Kd * vd
            acc += fs + fd + ft
        acc += V(0., -Grav, 0.)
        vel[i] += acc * Dt
    for i in pos:
        cond = (pos[i] < -1 and vel[i] < 0) or (pos[i] > 1 and vel[i] > 0)
        vel[i] = -0.1 * vel[i] if cond else vel[i]
        pos[i] += vel[i] * Dt

def step():
    for i in range(32):
        substep()

scene = tina.Scene()
model = tina.SimpleParticles(radius=R)
scene.add_object(model)

reset()
gui = ti.GUI()
while gui.running:
    scene.input(gui)
    if not gui.is_pressed(gui.SPACE): step()
    if gui.is_pressed('r'): reset()
    model.set_particles(pos.to_numpy())
    scene.render()
    gui.set_image(scene.img)
    gui.show()
