import taichi as ti
import numpy as np
import tina

ti.init(arch=ti.gpu)

### Parameters

N = 128
NN = N, N
W = 1
L = W / N
gravity = 0.5
stiffness = 1600
ball_radius = 0.4
damping = 2
steps = 30
dt = 5e-4

### Physics

x = ti.Vector.field(3, float, NN)
v = ti.Vector.field(3, float, NN)
ball_pos = ti.Vector.field(3, float, ())
ball_vel = ti.Vector.field(3, float, ())


@ti.kernel
def init():
    ball_pos[None] = ti.Vector([0.0, +0.0, 0.0])
    for i in ti.grouped(x):
        m, n = (i + 0.5) * L - 0.5
        x[i] = ti.Vector([m, 0.6, n])
        v[i] = ti.Vector([0.0, 0.0, 0.0])


links = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
links = [ti.Vector(_) for _ in links]

@ti.kernel
def substep():
    ball_pos[None] += ball_vel[None] * dt

    for i in ti.grouped(x):
        acc = x[i] * 0
        for d in ti.static(links):
            disp = x[min(max(i + d, 0), ti.Vector(NN) - 1)] - x[i]
            length = L * float(d).norm()
            acc += disp * (disp.norm() - length) / length**2
        v[i] += stiffness * acc * dt

    for i in ti.grouped(x):
        v[i].y -= gravity * dt
        dp = x[i] - ball_pos[None]
        dp2 = dp.norm_sqr()
        if dp2 <= ball_radius**2:
            # a fun execise left for you: add angular velocity to the ball?
            # it should drag the cloth around with a specific friction rate.
            dv = v[i] - ball_vel[None]
            NoV = dv.dot(dp)
            if NoV < 0:
                v[i] -= NoV * dp / dp2

    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)
        x[i] += dt * v[i]


### Rendering GUI

# Hint: remove ibl=True if you find it compiles too slow...
scene = tina.Scene((1024, 768), smoothing=True, texturing=True, ibl=True)

mesh = tina.MeshNoCulling(tina.MeshGrid((N, N)))
ball = tina.MeshTransform(tina.PrimitiveMesh.sphere())

cloth = tina.PBR(basecolor=tina.ChessboardTexture(size=0.2))
metal = tina.PBR(basecolor=[1.0, 0.9, 0.8], metallic=0.8, roughness=0.4)

scene.add_object(mesh, cloth)
scene.add_object(ball, metal)


gui = ti.GUI('Mass Spring', scene.res, fast_gui=True)
scene.init_control(gui,
        center=[0.0, 0.0, 0.0],
        theta=np.pi / 2 - np.radians(30),
        radius=1.5)

init()

print('[Hint] Press ASWD to move the ball, R to reset')
while gui.running:
    scene.input(gui)

    if not gui.is_pressed(gui.SPACE):
        for i in range(steps):
            substep()

    if gui.is_pressed('r'):
        init()

    if gui.is_pressed('w'):
        ball_vel[None] = (0, +1, 0)
    elif gui.is_pressed('s'):
        ball_vel[None] = (0, -1, 0)
    elif gui.is_pressed('a'):
        ball_vel[None] = (-1, 0, 0)
    elif gui.is_pressed('d'):
        ball_vel[None] = (+1, 0, 0)
    else:
        ball_vel[None] = (0, 0, 0)

    mesh.pos.copy_from(x)
    ball.set_transform(tina.translate(ball_pos[None].value) @ tina.scale(ball_radius))

    scene.render()
    gui.set_image(scene.img)
    gui.show()
