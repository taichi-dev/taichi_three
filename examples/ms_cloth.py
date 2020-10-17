import taichi as ti
import taichi_glsl as tl
import taichi_three as t3
import numpy as np

ti.init(arch=ti.cpu)

### Parameters

N = 128
NN = N, N
W = 1
L = W / N
gravity = 0.5
stiffness = 1600
ball_pos = tl.vec(+0.0, +0.2, -0.0)
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
        x[i] = tl.vec((i + 0.5) * L - 0.5, 0.8).xzy


links = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
links = [tl.vec(*_) for _ in links]


@ti.kernel
def substep():
    for i in ti.grouped(x):
        acc = x[i] * 0
        for d in ti.static(links):
            disp = x[tl.clamp(i + d, 0, tl.vec(*NN) - 1)] - x[i]
            length = L * float(d).norm()
            acc += disp * (disp.norm() - length) / length**2
        v[i] += stiffness * acc * dt
    for i in ti.grouped(x):
        v[i].y -= gravity * dt
        v[i] = tl.ballBoundReflect(x[i], v[i], ball_pos, ball_radius, 6)
    for i in ti.grouped(x):
        v[i] *= ti.exp(-damping * dt)
        x[i] += dt * v[i]


### Rendering GUI

scene = t3.Scene()
camera = t3.Camera()
camera.ctl = t3.CameraCtl(pos=[0, 0.8, -1.1], target=[0, 0.25, 0])
scene.add_camera(camera)
light = t3.Light(dir=[0.4, -1.5, 1.8])
scene.add_light(light)

mesh = t3.MeshGrid((N, N))
model = t3.Model(t3.QuadToTri(mesh))
model.material = t3.Material(t3.CookTorrance(color=t3.Texture('assets/cloth.jpg')))
scene.add_model(model)

sphere = t3.Model(t3.Mesh.from_obj('assets/sphere.obj'))
scene.add_model(sphere)


@ti.kernel
def update_display():
    for i in ti.grouped(x):
        mesh.pos[i] = x[i]


init()
sphere.L2W[None] = t3.translate(ball_pos) @ t3.scale(ball_radius)
with ti.GUI('Mass Spring', camera.res) as gui:
    while gui.running and not gui.get_event(gui.ESCAPE):
        if not gui.is_pressed(gui.SPACE):
            for i in range(steps):
                substep()
        if gui.is_pressed('r'):
            init()
        update_display()

        camera.from_mouse(gui)

        scene.render_shadows()
        scene.render()
        gui.set_image(camera.img)
        gui.show()
