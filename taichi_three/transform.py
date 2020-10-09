import taichi as ti
import taichi_glsl as ts
from .common import *
import math


def rotateX(angle):
    return ti.Matrix([
            [1,             0,              0, 0],
            [0, ti.cos(angle), -ti.sin(angle), 0],
            [0, ti.sin(angle),  ti.cos(angle), 0],
            [0,             0,              0, 1],
           ])

def rotateY(angle):
    return ti.Matrix([
            [ ti.cos(angle), 0, ti.sin(angle), 0],
            [             0, 1,             0, 0],
            [-ti.sin(angle), 0, ti.cos(angle), 0],
            [             0, 0,             0, 1],
           ])

def rotateZ(angle):
    return ti.Matrix([
            [ti.cos(angle), -ti.sin(angle), 0, 0],
            [ti.sin(angle),  ti.cos(angle), 0, 0],
            [            0,              0, 1, 0],
            [            0,              0, 0, 1],
           ])


def rotateAxis(axis, angle):
    if not isinstance(axis, ti.Matrix):
        axis = ti.Vector(axis)
    axis = axis.normalized()
    # FIXME: https://www.cnblogs.com/graphics/archive/2012/08/10/2627458.html
    return axis[0] * rotateX(angle) + axis[1] * rotateY(angle) + axis[2] * rotateZ(angle)


def transform(linear, offset):
    if not isinstance(linear, ti.Matrix):
        linear = ti.Matrix(linear)
    return ti.Matrix([
            [linear[0, 0], linear[0, 1], linear[0, 2], offset[0]],
            [linear[1, 0], linear[1, 1], linear[1, 2], offset[1]],
            [linear[2, 0], linear[2, 1], linear[2, 2], offset[2]],
            [           0,            0,            0,         1],
           ])


def translate(x, y, z):
    return ti.Matrix([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
           ])


def scale(x, y=None, z=None):
    y = y or x
    z = z or x
    return ti.Matrix([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1],
           ])


def translateScaleRotate(t, s, r, a):
    return rotateAxis(r, a) @ scale(s[0], s[1], s[2]) @ translate(t[0], t[1], t[2])


@ti.data_oriented
class Affine(ts.TaichiClass, AutoInit):
    @property
    def matrix(self):
        return self.entries[0]

    @property
    def offset(self):
        return self.entries[1]

    @classmethod
    def _field(cls, shape=None):
        return ti.Matrix.field(3, 3, float, shape), ti.Vector.field(3, float, shape)

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
        inv_mat = self.matrix.inverse()
        return Affine(inv_mat, -inv_mat @ self.offset)

    @ti.func
    def transpose(self):
        inv_mat = self.matrix.transpose()
        return Affine(inv_mat, -inv_mat @ self.offset)

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
