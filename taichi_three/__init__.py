'''
Taichi THREE
============

Taichi THREE is an extension library of the `Taichi Programming Language <https://github.com/taichi-dev/taichi>`_, that helps rendering your 3D scenes into nice-looking 2D images to display in GUI.
'''

__version__ = (0, 0, 7)
__author__ = '彭于斌 <1931127624@qq.com>'

import taichi as ti
import taichi_glsl as ts

print(f'[Tai3D] version {".".join(map(str, __version__))}')
print(f'[Tai3D] Inputs are welcomed at https://github.com/taichi-dev/taichi_three')

from .scene import *
from .model import *
from .scatter import *
from .camera import *
from .loader import *
from .light import *
from .raycast import *
from .objedit import *

print(f'[Tai3D] Camera control hints: LMB to orbit, MMB to move, RMB to scale')

from taichi import GUI, Vector, Matrix, kernel, func, random, init, reset, imread, imwrite, cpu, gpu
from taichi_glsl import sin, cos, tan, asin, acos, atan, bilerp, vec, vec2, vec3, vec4, isnan
from taichi_glsl import clamp, smoothstep, mix, sign, floor, ceil, fract, reflect, refract