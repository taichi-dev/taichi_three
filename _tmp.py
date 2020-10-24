import numblend as nb
import taichi as ti
import numpy as np
import bpy


print('=================')
nb.init()
ti.init(ti.cpu)

bpy.context.scene.frame_current = 0
nb.delete_mesh('point_cloud')
nb.delete_object('point_cloud')
mesh = nb.new_mesh('point_cloud')
nb.new_object('point_cloud', mesh)


@ti.data_oriented
class Solver:
    def __init__(self, N=32, dt=0.01):
        self.N = N
        self.dt = dt
        self.x = ti.Vector.field(3, float, N)
        self.v = ti.Vector.field(3, float, N)

    @ti.kernel
    def init(self):
        for i in self.x:
            self.x[i] = ti.Vector([ti.random() for i in range(3)]) * 2 - 1
            self.v[i] = ti.Vector([ti.random() for i in range(3)]) * 2 - 1

    @ti.kernel
    def substep(self):
        for i in self.x:
            self.x[i] += self.v[i] * self.dt


@ti.data_oriented
class Voxelizer:
    def __init__(self, N=128, min=-1, max=1):
        self.N = N
        self.min = ti.Vector([min for i in range(3)])
        self.max = ti.Vector([max for i in range(3)])
        self.f = ti.field(float)
        ti.root.dense(ti.ijk, N).place(self.f)

    @ti.kernel
    def clear(self):
        for i, j, k in self.f:
            self.f[i, j, k] = 0

    @ti.kernel
    def voxelize_particles(self, x: ti.template()):
        for i in x:
            p = (x[i] - self.min) / (self.max - self.min)
            I = p * self.N
            if not all(0 <= I < self.N):
                continue
            self.f[I] = 2

    @ti.kernel
    def _to_voxel_particles(self, out: ti.ext_arr()) -> int:
        len = 0
        for i, j, k in self.f:
            I = ti.Vector([i, j, k]) 
            if self.f[I] >= 1:
                n = ti.atomic_add(len, 1)
                p = I / self.N
                x = p * (self.max - self.min) + self.min
                for t in ti.static(range(3)):
                    out[n, t] = x[t]
        return len

    def to_voxel_particles(self):
        out = np.zeros((self.N**3, 3))
        len = self._to_voxel_particles(out)
        return out[:len]


@ti.data_oriented
class _GridCell:
    def __init__(self):
        self.p = [ti.expr_init(ti.Vector.zero(int, 3)) for t in range(8)]
        self.val = [ti.expr_init(0.0) for t in range(8)]

    @staticmethod
    @ti.func
    def make(data: ti.template(), i, j, k):
        grid = _GridCell()
        grid.p[0].x = i;  grid.p[0].y = j;  grid.p[0].z = k;  grid.val[0] = data[i,j,k];
        grid.p[1].x = i+1;grid.p[1].y = j;  grid.p[1].z = k;  grid.val[1] = data[i+1,j,k];
        grid.p[2].x = i+1;grid.p[2].y = j+1;grid.p[2].z = k;  grid.val[2] = data[i+1,j+1,k];
        grid.p[3].x = i;  grid.p[3].y = j+1;grid.p[3].z = k;  grid.val[3] = data[i,j+1,k];
        grid.p[4].x = i;  grid.p[4].y = j;  grid.p[4].z = k+1;grid.val[4] = data[i,j,k+1];
        grid.p[5].x = i+1;grid.p[5].y = j;  grid.p[5].z = k+1;grid.val[5] = data[i+1,j,k+1];
        grid.p[6].x = i+1;grid.p[6].y = j+1;grid.p[6].z = k+1;grid.val[6] = data[i+1,j+1,k+1];
        grid.p[7].x = i;  grid.p[7].y = j+1;grid.p[7].z = k+1;grid.val[7] = data[i,j+1,k+1];

    @ti.func
    def vertex_interp(self, p01 p2, v1, v2):
        k = (1 - v1) / (v2 - v1)
        p = p1 + k * (p2 - p1)
        return p

    @ti.func
    def polygonise(self, triangles: ti.template()):
        numOftriangle = 0;
        cubeIndex = 0;
        vertlist = [ti.expr_init(ti.Vector.zero(float, 3)) for t in range(12)]  # 保存等值面与立方体各边相交的坐标
        # 确定那个顶点位于等值面内部
        if (grid.val[0] < 1): cubeIndex |= 1;
        if (grid.val[1] < 1): cubeIndex |= 2;
        if (grid.val[2] < 1): cubeIndex |= 4;
        if (grid.val[3] < 1): cubeIndex |= 8;
        if (grid.val[4] < 1): cubeIndex |= 16;
        if (grid.val[5] < 1): cubeIndex |= 32;
        if (grid.val[6] < 1): cubeIndex |= 64;
        if (grid.val[7] < 1): cubeIndex |= 128;
        # 异常：立方体所有顶点都在或者都不在等值面内部
        if (self.edgeTable[cubeIndex] != 0):
            # 确定等值面与立方体交点坐标
            if (edgeTable[cubeIndex] & 1):
                vertlist[0] = self.vertex_interp(grid.p[0],grid.p[1],grid.val[0],grid.val[1]);
            if (edgeTable[cubeIndex] & 2):
                vertlist[1] = self.vertex_interp(grid.p[1],grid.p[2],grid.val[1],grid.val[2]);
            if (edgeTable[cubeIndex] & 4):
                vertlist[2] = self.vertex_interp(grid.p[2],grid.p[3],grid.val[2],grid.val[3]);
            if (edgeTable[cubeIndex] & 8):
                vertlist[3] = self.vertex_interp(grid.p[3],grid.p[0],grid.val[3],grid.val[0]);
            if (edgeTable[cubeIndex] & 16):
                vertlist[4] = self.vertex_interp(grid.p[4],grid.p[5],grid.val[4],grid.val[5]);
            if (edgeTable[cubeIndex] & 32):
                vertlist[5] = self.vertex_interp(grid.p[5],grid.p[6],grid.val[5],grid.val[6]);
            if (edgeTable[cubeIndex] & 64):
                vertlist[6] = self.vertex_interp(grid.p[6],grid.p[7],grid.val[6],grid.val[7]);
            if (edgeTable[cubeIndex] & 128):
                vertlist[7] = self.vertex_interp(grid.p[7],grid.p[4],grid.val[7],grid.val[4]);
            if (edgeTable[cubeIndex] & 256):
                vertlist[8] = self.vertex_interp(grid.p[0],grid.p[4],grid.val[0],grid.val[4]);
            if (edgeTable[cubeIndex] & 512):
                vertlist[9] = self.vertex_interp(grid.p[1],grid.p[5],grid.val[1],grid.val[5]);
            if (edgeTable[cubeIndex] & 1024):
                vertlist[10] = self.vertex_interp(grid.p[2],grid.p[6],grid.val[2],grid.val[6]);
            if (edgeTable[cubeIndex] & 2048):
                vertlist[11] = self.vertex_interp(grid.p[3],grid.p[7],grid.val[3],grid.val[7]);
            //根据交点坐标确定三角形面片，并进行保存
            for (int i=0; triTable[cubeIndex][i] != -1; i+=3):
            tri[numOftriangle].p[0] = vertlist[ triTable[cubeIndex][i  ] ];
            tri[numOftriangle].p[1] = vertlist[ triTable[cubeIndex][i+1] ];
            tri[numOftriangle].p[2] = vertlist[ triTable[cubeIndex][i+2] ];
            numOftriangle++;

        


@ti.data_oriented
class Marcher(Voxelizer):
    def __init__(self, N=128, min=-1, max=1):
        super().__init__(N, min, max)

    def march(self):
        for i, j, k in self.f:
            grid = _GridCell.make(self.f, i, j, k)



solver = Solver()
marcher = Marcher(min=-2, max=2)


@nb.add_animation
def main():
    solver.init()
    for frame in range(250):
        solver.substep()
        marcher.voxelize_particles(solver.x)
        pos = marcher.to_voxel_particles()
        print(pos)
        yield nb.mesh_update(mesh, pos)


bpy.ops.screen.animation_play()