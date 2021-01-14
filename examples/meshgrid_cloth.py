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
ball_pos = ti.Vector([+0.0, +0.2, -0.0])
ball_radius = 0.4
damping = 2
steps = 30
dt = 5e-4

### Physics

x = ti.Vector.field(3, float, NN)
v = ti.Vector.field(3, float, NN)


@ti.kernel
def init():
    for i in ti.grouped(x):
        u, v = (i + 0.5) * L - 0.5
        x[i] = ti.Vector([u, 0.8, v])


links = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
links = [ti.Vector(_) for _ in links]

@ti.kernel
def substep():
    for i in ti.grouped(x):
        acc = x[i] * 0
        for d in ti.static(links):
            disp = x[min(max(i + d, 0), ti.Vector(NN) - 1)] - x[i]
            length = L * float(d).norm()
            acc += disp * (disp.norm() - length) / length**2
        v[i] += stiffness * acc * dt
    for i in ti.grouped(x):
        v[i].y -= gravity * dt
        disp = x[i] - ball_pos
        disp2 = disp.norm_sqr()
        if disp2 <= ball_radius**2:
            NoV = v[i].dot(disp)
            if NoV < 0: v[i] -= NoV * disp / disp2
    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)
        x[i] += dt * v[i]


### Rendering GUI

scene = tina.Scene((1024, 768), smoothing=True, texturing=True, ibl=True)

mesh = tina.MeshNoCulling(tina.MeshGrid((N, N)))
ball = tina.MeshTransform(tina.PrimitiveMesh.sphere())

cloth = tina.PBR(basecolor=tina.ChessboardTexture(size=0.2))
metal = tina.PBR(basecolor=[1.0, 0.9, 0.8], metallic=0.8, roughness=0.4)

scene.add_object(mesh, cloth)
scene.add_object(ball, metal)


gui = ti.GUI('Mass Spring', scene.res, fast_gui=True)
scene.init_control(gui,
        center=ball_pos,
        theta=np.pi / 2 - np.radians(30),
        radius=1.5)

#scene.lighting.clear_lights()
#scene.lighting.add_light(dir=[0, 1, 1], color=[0.9, 0.9, 0.9])
#scene.lighting.set_ambient_light([0.1, 0.1, 0.1])

ball.set_transform(tina.translate(ball_pos) @ tina.scale(ball_radius))

init()

while gui.running:
    scene.input(gui)

    if not gui.is_pressed(gui.SPACE):
        for i in range(steps):
            substep()

    if gui.is_pressed('r'):
        init()

    mesh.pos.copy_from(x)

    scene.render()
    gui.set_image(scene.img)
    gui.show()
