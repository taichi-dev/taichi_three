import taichi as ti
import taichi_glsl as ts
from .common import *
import math


def crossProduct(a, b):
    x, y, z = a
    u, v, w = b
    return y * w - z * v, z * u - w * x, x * v - y * u

def dotProduct(a, b):
    x, y, z = a
    u, v, w = b
    return x * u + y * v + z * w

def vectorAdd(a, b):
    x, y, z = a
    u, v, w = b
    return x + u, y + v, z + w

def vectorSub(a, b):
    x, y, z = a
    u, v, w = b
    return x - u, y - v, z - w

def vectorMul(a, k):
    x, y, z = a
    return x * k, y * k, z * k


def rotationX(angle):
    return [
            [1,               0,                0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle),  math.cos(angle)],
           ]

def rotationY(angle):
    return [
            [ math.cos(angle), 0, math.sin(angle)],
            [               0, 1,               0],
            [-math.sin(angle), 0, math.cos(angle)],
           ]

def rotationZ(angle):
    return [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle),  math.cos(angle), 0],
            [              0,                0, 1],
           ]


@ti.data_oriented
class Affine(ts.TaichiClass, AutoInit):
    @property
    def matrix(self):
        return self.entries[0]

    @property
    def offset(self):
        return self.entries[1]

    @classmethod
    def _var(cls, shape=None):
        return ti.Matrix(3, 3, ti.f32, shape), ti.Vector.var(3, ti.f32, shape)

    @ti.func
    def loadIdentity(self):
        self.matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.offset = [0, 0, 0]

    @ti.kernel
    def _init(self):
        self.loadIdentity()

    @ti.func
    def __matmul__(self, other):
        return self.matrix @ other + self.offset

    @ti.func
    def inverse(self):
        return Affine(self.matrix.inverse(), -self.offset)

    def variable(self):
        return Affine(self.matrix.variable(), self.offset.variable())

    def loadOrtho(self, fwd=[0, 0, 1], up=[0, 1, 0]):
        # fwd = target - pos
        # fwd = fwd.normalized()
        fwd_len = math.sqrt(sum(x**2 for x in fwd))
        fwd = [x / fwd_len for x in fwd]
        # right = fwd.cross(up)
        right = [
                fwd[2] * up[1] - fwd[1] * up[2],
                fwd[0] * up[2] - fwd[2] * up[0],
                fwd[1] * up[0] - fwd[0] * up[1],
                ]
        # right = right.normalized()
        right_len = math.sqrt(sum(x**2 for x in right))
        right = [x / right_len for x in right]
        # up = right.cross(fwd)
        up = [
             right[2] * fwd[1] - right[1] * fwd[2],
             right[0] * fwd[2] - right[2] * fwd[0],
             right[1] * fwd[0] - right[0] * fwd[1],
             ]

        # trans = ti.Matrix.cols([right, up, fwd])
        trans = [right, up, fwd]
        trans = [[trans[i][j] for i in range(3)] for j in range(3)]
        self.matrix[None] = trans

    def from_mouse(self, mpos):
        if isinstance(mpos, ti.GUI):
            if mpos.is_pressed(ti.GUI.LMB):
                mpos = mpos.get_cursor_pos()
            else:
                mpos = (0, 0)
        a, t = mpos
        if a != 0 or t != 0:
            a, t = a * math.tau - math.pi, t * math.pi - math.pi / 2
            c = math.cos(t)
            self.loadOrtho(fwd=[c * math.sin(a), math.sin(t), c * math.cos(a)])


@ti.data_oriented
class Camera(AutoInit):
    ORTHO = 'Orthogonal'
    TAN_FOV = 'Tangent Perspective'
    COS_FOV = 'Cosine Perspective'

    def __init__(self, pos=[0, 0, -2], target=[0, 0, 0], up=[0, 1, 0]):
        self.trans = ti.Matrix(3, 3, ti.f32, ())
        self.pos = ti.Vector(3, ti.f32, ())
        self.target = ti.Vector(3, ti.f32, ())
        self.type = self.TAN_FOV
        self.fov = 25
        # python scope camera transformations
        self.pos_py = pos
        self.target_py = target
        self.up_py = up
        # mouse position for camera control
        self.mpos = (0, 0)

    '''
    NOTE: taichi_three uses a LEFT HANDED coordinate system.
    that is, the +Z axis points FROM the camera TOWARDS the scene,
    with X, Y being device coordinates
    '''
    def set(self, pos=None, target=None, up=None):
        pos = self.pos_py if pos is None else pos
        target = self.target_py if target is None else target
        up = self.up_py if up is None else up
        
        # fwd = target - pos
        fwd = [target[i] - pos[i] for i in range(3)]
        # fwd = fwd.normalized()
        fwd_len = math.sqrt(sum(x**2 for x in fwd))
        fwd = [x / fwd_len for x in fwd]
        # right = fwd.cross(up) 
        right = [
                fwd[2] * up[1] - fwd[1] * up[2],
                fwd[0] * up[2] - fwd[2] * up[0],
                fwd[1] * up[0] - fwd[0] * up[1],
                ]
        # right = right.normalized()
        right_len = math.sqrt(sum(x**2 for x in right))
        right = [x / right_len for x in right]
        # up = right.cross(fwd)
        up = [
             right[2] * fwd[1] - right[1] * fwd[2],
             right[0] * fwd[2] - right[2] * fwd[0],
             right[1] * fwd[0] - right[0] * fwd[1],
             ]

        # trans = ti.Matrix.cols([right, up, fwd])
        trans = [right, up, fwd]
        trans = [[trans[i][j] for i in range(3)] for j in range(3)]
        self.trans[None] = trans
        self.pos[None] = pos
        self.target[None] = target
        self.pos_py = pos
        self.target_py = target

    def _init(self):
        self.set()

    def from_mouse(self, gui):
        if gui.is_pressed(ti.GUI.LMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.orbit((mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]),
                    pov=gui.is_pressed(ti.GUI.CTRL))
            self.mpos = mpos
        elif gui.is_pressed(ti.GUI.RMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.zoom_by_mouse(mpos, (mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]))
            self.mpos = mpos
        elif gui.is_pressed(ti.GUI.MMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.pan((mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]))
            self.mpos = mpos
        else:
            if gui.event and gui.event.key == ti.GUI.WHEEL:
                # one mouse wheel unit is (0, 120)
                self.zoom(-gui.event.delta[1] / 1200)
                gui.event = None
            mpos = (0, 0)
        self.mpos = mpos


    def orbit(self, delta, sensitivity=5, pov=False):
        ds, dt = delta
        if ds != 0 or dt != 0:
            dis = math.sqrt(sum((self.target_py[i] - self.pos_py[i]) ** 2 for i in range(3)))
            fov = math.radians(self.fov)
            ds, dt = ds * fov * sensitivity, dt * fov * sensitivity
            newdir = ts.vec3(ds, dt, 1).normalized()
            newdir = [sum(self.trans[None][i, j] * newdir[j] for j in range(3))\
                        for i in range(3)]
            if pov:
                newtarget = [self.pos_py[i] + dis * newdir[i] for i in range(3)]
                self.set(target=newtarget)
            else:
                newpos = [self.target_py[i] - dis * newdir[i] for i in range(3)]
                self.set(pos=newpos)

    def zoom_by_mouse(self, pos, delta, sensitivity=3):
        ds, dt = delta
        if ds != 0 or dt != 0:
            z = math.sqrt(ds ** 2 + dt ** 2) * sensitivity
            if (pos[0] - 0.5) * ds + (pos[1] - 0.5) * dt > 0:
                z *= -1
            self.zoom(z)
    
    def zoom(self, z):
        newpos = [(1 + z) * self.pos_py[i] - z * self.target_py[i] for i in range(3)]
        newtarget = [z * self.pos_py[i] + (1 - z) * self.target_py[i] for i in range(3)]
        self.set(pos=newpos, target=newtarget)

    def pan(self, delta, sensitivity=3):
        ds, dt = delta
        if ds != 0 or dt != 0:
            dis = math.sqrt(sum((self.target_py[i] - self.pos_py[i]) ** 2 for i in range(3)))
            fov = math.radians(self.fov)
            ds, dt = ds * fov * sensitivity, dt * fov * sensitivity
            newdir = ts.vec3(-ds, -dt, 1).normalized()
            newdir = [sum(self.trans[None][i, j] * newdir[j] for j in range(3))\
                        for i in range(3)]
            newtarget = [self.pos_py[i] + dis * newdir[i] for i in range(3)]
            newpos = [self.pos_py[i] + newtarget[i] - self.target_py[i] for i in range(3)]
            self.set(pos=newpos, target=newtarget)


    @ti.func
    def trans_pos(self, pos):
        return self.trans[None] @ pos + self.pos[None]

    @ti.func
    def trans_dir(self, pos):
        return self.trans[None] @ pos

    @ti.func
    def untrans_pos(self, pos):
        return self.trans[None].inverse() @ (pos - self.pos[None])

    @ti.func
    def untrans_dir(self, pos):
        return self.trans[None].inverse() @ pos

    @ti.func
    def generate(self, coor):
        fov = ti.static(math.radians(self.fov))
        tan_fov = ti.static(math.tan(fov))

        orig = ts.vec3(0.0)
        dir  = ts.vec3(0.0, 0.0, 1.0)

        if ti.static(self.type == self.ORTHO):
            orig = ts.vec3(coor, 0.0)
        elif ti.static(self.type == self.TAN_FOV):
            uv = coor * fov
            dir = ts.normalize(ts.vec3(uv, 1))
        elif ti.static(self.type == self.COS_FOV):
            uv = coor * fov
            dir = ts.vec3(ti.sin(uv), ti.cos(uv.norm()))

        orig = self.trans_pos(orig)
        dir = self.trans_dir(dir)

        return orig, dir
