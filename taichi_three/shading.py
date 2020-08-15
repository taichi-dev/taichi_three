import taichi as ti
import taichi_glsl as ts
from .transform import *
import math

class Shading:
    def __init__(self, **kwargs):
        self.is_normal_map = False
        self.lambert = 0.58
        self.half_lambert = 0.04
        self.blinn_phong = 0.3
        self.phong = 0.0
        self.shineness = 10
        self.__dict__.update(kwargs)

    @ti.func
    def render_func(self, pos, normal, dir, light):
        color = ts.vec3(0.0)
        light_dir = light.get_dir(pos)
        shineness = self.shineness
        half_lambert = ts.dot(normal, light_dir) * 0.5 + 0.5
        lambert = max(0, ts.dot(normal, light_dir))
        blinn_phong = ts.dot(normal, ts.mix(light_dir, -dir, 0.5))
        blinn_phong = pow(max(blinn_phong, 0), shineness)
        refl_dir = ts.reflect(light_dir, normal)
        phong = -ts.dot(normal, refl_dir)
        phong = pow(max(phong, 0), shineness)

        strength = 0.0
        if ti.static(self.lambert != 0.0):
            strength += lambert * self.lambert
        if ti.static(self.half_lambert != 0.0):
            strength += half_lambert * self.half_lambert
        if ti.static(self.blinn_phong != 0.0):
            strength += blinn_phong * self.blinn_phong
        if ti.static(self.phong != 0.0):
            strength += phong * self.phong
        color = ts.vec3(strength)

        if ti.static(self.is_normal_map):
            color = normal * 0.5 + 0.5
        return color * light.get_color(pos)

    @ti.func
    def pre_process(self, color):
        blue = ts.vec3(0.00, 0.01, 0.05)
        orange = ts.vec3(1.19, 1.04, 0.98)
        return ti.sqrt(ts.mix(blue, orange, color))
