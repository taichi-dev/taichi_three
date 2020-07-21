import taichi as ti
import taichi_glsl as ts
from .common import *
import math


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
        # TODO: incorrect:
        return Affine(self.matrix.inverse(), -self.offset)

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
            mpos = mpos.get_cursor_pos()

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

    def __init__(self):
        self.trans = ti.Matrix(3, 3, ti.f32, ())
        self.pos = ti.Vector(3, ti.f32, ())
        self.intrinsic = ti.Matrix(3, 3, ti.f32, ())
        self.type = self.TAN_FOV
        self.fov = 25

    def set(self, pos=[0, 0, -2], target=[0, 0, 0], up=[0, 1, 0]):
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
    
    def set_intrinsic(self, fx = 1, fy = 1, cx = 0, cy = 0):
        self.intrinsic[None][0, 0] = fx
        self.intrinsic[None][0, 2] = cx
        self.intrinsic[None][1, 1] = fy
        self.intrinsic[None][1, 2] = cy
        self.intrinsic[None][2, 2] = 1.0

    def _init(self):
        self.set()
        self.set_intrinsic()

    def from_mouse(self, mpos, dis=3):
        if isinstance(mpos, ti.GUI):
            mpos = mpos.get_cursor_pos()

        a, t = mpos
        if a != 0 or t != 0:
            a, t = a * math.tau - math.pi, t * math.pi - math.pi / 2
        d = dis * math.cos(t)
        self.set(pos=[d * math.sin(a), dis * math.sin(t), d * math.cos(a)])

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
    def uncook(self, pos):
        if ti.static(self.type == self.ORTHO):
            pos[0] *= self.intrinsic[None][0, 0] 
            pos[1] *= self.intrinsic[None][1, 1]
            pos[0] += self.intrinsic[None][0, 2]
            pos[1] += self.intrinsic[None][1, 2]
        else:
            pos = self.intrinsic[None] @ pos
            pos[0] /= pos[2]
            pos[1] /= pos[2]
        return ts.vec2(pos[0], pos[1])

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
