import numpy as np
import taichi as ti
import taichi_glsl as ts
from .model import *


class SimpleModel(Model):
    def __init__(self, n_vertices=4096 * 3, n_triangles=4096):
        self.tex = ti.Vector.field(3, float, n_vertices)
        super().__init__(n_triangles, n_vertices, n_vertices, n_vertices)
        self.n_vertices = ti.field(int, ())
        self.n_triangles = ti.field(int, ())

        self.gl = OpenGL(self._opengl_callback)

    @ti.func
    def render(self, camera):
        for i in range(self.n_triangles[None]):
            # assume all elements to be triangle
            render_triangle(self, camera, self.faces[i])

    @ti.func
    def pixel_shader(self, pos, color):
        return dict(img=color, pos=pos)

    @ti.func
    def vertex_shader(self, pos, color, normal, tangent, bitangent):
        return pos, color

    @ti.kernel
    def set_vertices(self, data: ti.ext_arr()):
        self.n_vertices[None] = min(ti.Expr(self.pos.shape[0]), data.shape[0])
        for i in range(self.n_vertices[None]):
            for j in ti.static(range(3)):
                self.pos[i][j] = data[i, 0, j]
                self.tex[i][j] = data[i, 1, j]
                self.nrm[i][j] = data[i, 2, j]

    @ti.kernel
    def set_triangles(self, data: ti.ext_arr()):
        self.n_triangles[None] = min(ti.Expr(self.faces.shape[0]), data.shape[0])
        for i in range(self.n_triangles[None]):
            for j, k in ti.static(ti.ndrange(3, 3)):
                self.faces[i][j, k] = data[i, j]

    def _opengl_callback(self, data, type):
        n = len(data) // 3
        faces = np.arange(0, n * 3, dtype=np.int32).reshape(n, 3)
        data = np.array(data, dtype=np.float32)
        self.set_vertices(data)
        self.set_triangles(faces)


@ti.data_oriented
class OpenGL:
    def __init__(self, callback):
        self.callback = callback
        self.color = (1, 1, 1)
        self.normal = (0, 0, 0)

    def End(self):
        data = np.array(self.data)
        self.callback(self.data, self.type)
        del self.data
        del self.type

    def Begin(self, type=None):
        self.data = []
        self.type = type

    def Vertex(self, x, y, z=0):
        data = (x, y, z), self.color, self.normal
        self.data.append(data)

    def Color(self, r, g, b):
        self.color = (r, g, b)

    def Normal(self, x, y, z):
        self.normal = (x, y, z)

    def Rect(self, a, b, c, d):
        self.Vertex(a, b, c)
        self.Vertex(b, c, d)