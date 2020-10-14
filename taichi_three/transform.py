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


def translate(x, y=None, z=None):
    if y is None or z is None:
        x, y, z = x
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