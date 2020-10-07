import taichi as ti
import taichi_glsl as ts
from .common import *
from .transform import *
import math


@ti.data_oriented
class FrameBuffer:
    def __init__(self, res=None):
        self.res = res or (512, 512)
        self.buffers = {}
        self.add_buffer('img', 3)
        self.add_buffer('idepth', 0)

    def add_buffer(self, name, dim, dtype=float):
        if dim == 0:
            buf = ti.field(dtype, self.res)
        else:
            buf = ti.Vector.field(dim, dtype, self.res)
        self.buffers[name] = buf

    def __getitem__(self, name):
        if isinstance(name, tuple):
            name = name[0]
        if name in self.buffers:
            return self.buffers[name]
        else:
            return dummy_expression()

    @ti.func
    def update(self, I, res: ti.template()):
        for k, v in ti.static(res.items()):
            self[k][I] = v

    @ti.func
    def clear_buffer(self):
        for I in ti.grouped(next(iter(self.buffers.values()))):
            for buf in ti.static(self.buffers.values()):
                buf[I] *= 0.0


# TODO: separate intrinsic to FrameBuffer, leave extrinsic to Camera?
@ti.data_oriented
class Camera(AutoInit):
    ORTHO = 'Orthogonal'
    TAN_FOV = 'Tangent Perspective' # rectilinear perspective
    COS_FOV = 'Cosine Perspective' # curvilinear perspective, see en.wikipedia.org/wiki/Curvilinear_perspective

    def __init__(self, res=None, fx=None, fy=None, cx=None, cy=None,
            pos=None, target=None, up=None, fov=None):
        self.res = res or (512, 512)
        self.fb = FrameBuffer(self.res)
        self.affine = Affine.field(())
        self.target = ti.Vector.field(3, ti.f32, ())
        self.intrinsic = ti.Matrix.field(3, 3, ti.f32, ())
        self.type = self.TAN_FOV
        self.fov = math.radians(fov or 30)

        minres = min(self.res)
        self.cx = cx or self.res[0] / 2
        self.cy = cy or self.res[1] / 2
        self.fx = fx or minres / (2 * math.tan(self.fov))
        self.fy = fy or minres / (2 * math.tan(self.fov))
        # python scope camera transformations
        self.pos_py = pos or [0, 0, -2]
        self.target_py = target or [0, 0, 0]
        self.trans_py = None
        self.up_py = up or [0, 1, 0]
        self.set(init=True)
        # mouse position for camera control
        self.mpos = (0, 0)

    @property
    def pos(self):
        return self.affine.offset

    @property
    def trans(self):
        return self.affine.matrix

    def set_intrinsic(self, fx=None, fy=None, cx=None, cy=None):
        # see http://ais.informatik.uni-freiburg.de/teaching/ws09/robotics2/pdfs/rob2-08-camera-calibration.pdf
        self.fx = fx or self.fx
        self.fy = fy or self.fy
        self.cx = cx or self.cx
        self.cy = cy or self.cy

    '''
    NOTE: taichi_three uses a LEFT HANDED coordinate system.
    that is, the +Z axis points FROM the camera TOWARDS the scene,
    with X, Y being device coordinates.
    '''
    def set(self, pos=None, target=None, up=None, init=False):
        pos = ti.Vector(pos or self.pos_py)
        target = ti.Vector(target or self.target_py)
        up = ti.Vector(up or self.up_py)
        fwd = (target - pos).normalized()
        right = up.cross(fwd).normalized()
        up = fwd.cross(right)
        trans = ti.Matrix([right.entries, up.entries, fwd.entries]).transpose()

        self.trans_py = [[trans[i, j] for j in range(3)] for i in range(3)]
        self.pos_py = pos.entries
        self.target_py = target.entries
        if not init:
            self.affine.offset[None] = self.pos_py
            self.affine.matrix[None] = self.trans_py
            self.target[None] = self.target_py

    def _init(self):
        self.affine.offset[None] = self.pos_py
        self.affine.matrix[None] = self.trans_py
        self.target[None] = self.target_py
        self.intrinsic[None][0, 0] = self.fx
        self.intrinsic[None][0, 2] = self.cx
        self.intrinsic[None][1, 1] = self.fy
        self.intrinsic[None][1, 2] = self.cy
        self.intrinsic[None][2, 2] = 1.0

    @property
    def img(self):
        return self.fb['img']

    def from_mouse(self, gui):
        is_alter_move = gui.is_pressed(ti.GUI.CTRL)
        if gui.is_pressed(ti.GUI.LMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.orbit((mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]),
                    pov=is_alter_move)
            self.mpos = mpos
        elif gui.is_pressed(ti.GUI.RMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.zoom_by_mouse(mpos, (mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]),
                        dolly=is_alter_move)
            self.mpos = mpos
        elif gui.is_pressed(ti.GUI.MMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.pan((mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]))
            self.mpos = mpos
        else:
            if gui.event and gui.event.key == ti.GUI.WHEEL:
                # one mouse wheel unit is (0, 120)
                self.zoom(-gui.event.delta[1] / 1200,
                    dolly=is_alter_move)
                gui.event = None
            mpos = (0, 0)
        self.mpos = mpos


    def orbit(self, delta, sensitivity=5, pov=False):
        ds, dt = delta
        if ds != 0 or dt != 0:
            dis = math.sqrt(sum((self.target_py[i] - self.pos_py[i]) ** 2 for i in range(3)))
            fov = self.fov
            ds, dt = ds * fov * sensitivity, dt * fov * sensitivity
            newdir = ts.vec3(ds, dt, 1).normalized()
            newdir = [sum(self.affine.matrix[None][i, j] * newdir[j] for j in range(3))\
                        for i in range(3)]
            if pov:
                newtarget = [self.pos_py[i] + dis * newdir[i] for i in range(3)]
                self.set(target=newtarget)
            else:
                newpos = [self.target_py[i] - dis * newdir[i] for i in range(3)]
                self.set(pos=newpos)

    def zoom_by_mouse(self, pos, delta, sensitivity=3, dolly=False):
        ds, dt = delta
        if ds != 0 or dt != 0:
            z = math.sqrt(ds ** 2 + dt ** 2) * sensitivity
            if (pos[0] - 0.5) * ds + (pos[1] - 0.5) * dt > 0:
                z *= -1
            self.zoom(z, dolly)
    
    def zoom(self, z, dolly=False):
        newpos = [(1 + z) * self.pos_py[i] - z * self.target_py[i] for i in range(3)]
        if dolly:
            newtarget = [z * self.pos_py[i] + (1 - z) * self.target_py[i] for i in range(3)]
            self.set(pos=newpos, target=newtarget)
        else:
            self.set(pos=newpos)

    def pan(self, delta, sensitivity=3):
        ds, dt = delta
        if ds != 0 or dt != 0:
            dis = math.sqrt(sum((self.target_py[i] - self.pos_py[i]) ** 2 for i in range(3)))
            fov = self.fov
            ds, dt = ds * fov * sensitivity, dt * fov * sensitivity
            newdir = ts.vec3(-ds, -dt, 1).normalized()
            newdir = [sum(self.affine.matrix[None][i, j] * newdir[j] for j in range(3))\
                        for i in range(3)]
            newtarget = [self.pos_py[i] + dis * newdir[i] for i in range(3)]
            newpos = [self.pos_py[i] + newtarget[i] - self.target_py[i] for i in range(3)]
            self.set(pos=newpos, target=newtarget)

    @ti.func
    def trans_pos(self, pos):
        return self.affine[None] @ pos

    @ti.func
    def trans_dir(self, pos):
        return self.affine.matrix[None] @ pos

    @ti.func
    def untrans_pos(self, pos):
        return self.affine[None].transpose() @ pos

    @ti.func
    def untrans_dir(self, pos):
        return self.affine.matrix[None].transpose() @ pos

    @ti.func
    def cook(self, pos, translate=True):
        if ti.static(self.type == self.ORTHO):
            if ti.static(translate):
                pos[0] -= self.intrinsic[None][0, 2]
                pos[1] -= self.intrinsic[None][1, 2]
            pos[0] /= self.intrinsic[None][0, 0] 
            pos[1] /= self.intrinsic[None][1, 1]

        elif ti.static(self.type == self.TAN_FOV):
            pos[0] *= abs(pos[2])
            pos[1] *= abs(pos[2])
            pos = self.intrinsic[None].inverse() @ pos

        else:
            raise NotImplementedError("Curvilinear projection matrix not implemented!")

        return pos
    
    @ti.func
    def uncook(self, pos, translate: ti.template() = True):
        if ti.static(self.type == self.ORTHO):
            pos[0] *= self.intrinsic[None][0, 0] 
            pos[1] *= self.intrinsic[None][1, 1]
            if ti.static(translate):
                pos[0] += self.intrinsic[None][0, 2]
                pos[1] += self.intrinsic[None][1, 2]

        elif ti.static(self.type == self.TAN_FOV):
            if ti.static(translate):
                pos = self.intrinsic[None] @ pos
            else:
                pos[0] *= self.intrinsic[None][0, 0]
                pos[1] *= self.intrinsic[None][1, 1]

            pos[0] /= abs(pos[2])
            pos[1] /= abs(pos[2])

        else:
            raise NotImplementedError("Curvilinear projection matrix not implemented!")

        return ts.vec2(pos[0], pos[1])

    def export_intrinsic(self):
        import numpy as np
        intrinsic = np.zeros((3, 3))
        intrinsic[0, 0] = self.fx
        intrinsic[1, 1] = self.fy
        intrinsic[0, 2] = self.cx
        intrinsic[1, 2] = self.cy
        intrinsic[2, 2] = 1
        return intrinsic

    def export_extrinsic(self):
        import numpy as np
        trans = np.array(self.trans_py)
        pos = np.array(self.pos_py)
        extrinsic = np.zeros((3, 4))

        trans = np.transpose(trans)
        for i in range(3):
            for j in range(3):
                extrinsic[i][j] = trans[i, j]
        pos = -trans @ pos
        for i in range(3):
            extrinsic[i][3] = pos[i]
        return extrinsic