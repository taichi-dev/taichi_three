'''
Taichi THREE
============

Taichi THREE is an extension library of the `Taichi Programming Language <https://github.com/taichi-dev/taichi>`_, that helps rendering your 3D scenes into nice-looking 2D images to display in GUI.
'''

__version__ = (0, 0, 8)
__author__ = '彭于斌 <1931127624@qq.com>'
__license__ = 'MIT'

import taichi as ti
import taichi_glsl as ts
import numpy as np

print(f'[Tai3D] version {".".join(map(str, __version__))}')
print(f'[Tai3D] Documentation: https://t3.142857.red')
print(f'[Tai3D] GitHub: https://github.com/taichi-dev/taichi_three')

from .scene import *
from .model import *
from .interme import *
from .scatter import *
from .camera import *
from .buffer import *
from .loader import *
from .light import *
from .raycast import *
from .skybox import *
from .objedit import *

print(f'[Tai3D] Camera control hints: LMB to orbit, MMB to move, RMB to scale')

from taichi import GUI, Vector, Matrix, kernel, func, random, init, reset, imread, imwrite, cpu, gpu
from taichi_glsl import clamp, smoothstep, mix, sign, sqrt, floor, ceil, fract, reflect, refract
from taichi_glsl import sin, cos, tan, asin, acos, atan, bilerp, vec, vec2, vec3, vec4, isnan
from math import radians, degrees, pi, tau

@ti.python_scope
def get_time():
    import time
    return time.time() % 65536

def RGB(r, g, b):
    return ti.Vector([r, g, b])

def Vec(*args):
    return ti.Vector(args)

ti.reset()