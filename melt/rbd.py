from tina.advans import *

N = 128
pos = ti.Vector.field(3, float, N, needs_grad=True)
U = ti.field(float, (), needs_grad=True)

@ti.materialize_callback
@ti.kernel
def reset():
    for i in pos:
        pos[i] = V(ti.random(), ti.random())

@ti.pyfunc
def sdPlane(a, b, p):
    n = (b - a).Yx
    s = (p - a).dot(n)
    return s / n.norm()

@ti.pyfunc
def sdCube(a, s, p):
    s1 = sdPlane(a + V(-s, -s), a + V(-s, +s), p)
    s2 = sdPlane(a + V(-s, +s), a + V(+s, +s), p)
    s3 = sdPlane(a + V(+s, +s), a + V(+s, -s), p)
    s4 = sdPlane(a + V(+s, -s), a + V(-s, -s), p)
    return max(s1, s2, s3, s4)

@ti.kernel
def calcEnergy():
    for i in range(N):
        for j in range(i + 1, N):
            dis = (pos[i] - pos[j]).norm()
            U[None] += max(0, 0.1 - dis)

@ti.kernel
def stepForward():
    for i in range(N):
        pos[i] -= pos.grad[i] * 0.001


gui = ti.GUI()
while gui.running and not gui.get_event(gui.ESCAPE):
    with ti.Tape(U):
        calcEnergy()
    stepForward()
    print('U =', U[None])
    gui.circles(pos.to_numpy(), radius=4)
    gui.show()
