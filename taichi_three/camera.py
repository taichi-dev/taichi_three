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

    @ti.func
    def idepth_fixp(self, z):
        if ti.static(ti.core.is_integral(self['idepth'].dtype)):
            return int(2**24 * z)
        else:
            return z

    @ti.func
    def atomic_depth(self, X, depth):
        idepth = self.idepth_fixp(1 / depth)
        return idepth < ti.atomic_max(self['idepth'][X], idepth)

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
    def update(self, I, data: ti.template()):
        for k, v in ti.static(data.items()):
            self[k][I] = v

    def fetchpixelinfo(self, name, pos):
        if name in self.buffers:
            I = int(pos[0] * self.res[0]), int(pos[1] * self.res[1])
            return self[name][I].value
        else:
            return None

    @ti.func
    def clear_buffer(self):
        for I in ti.grouped(next(iter(self.buffers.values()))):
            for buf in ti.static(self.buffers.values()):
                buf[I] *= 0


@ti.data_oriented
class Camera:
    ORTHO = 'Orthogonal'
    TAN_FOV = 'Tangent Perspective' # rectilinear perspective
    COS_FOV = 'Cosine Perspective' # curvilinear perspective, see en.wikipedia.org/wiki/Curvilinear_perspective

    def __init__(self, res=None):
        self.res = res or (512, 512)
        self.fb = FrameBuffer(self.res)
        self.L2W = ti.Matrix.field(4, 4, float, ())
        self.intrinsic = ti.Matrix.field(3, 3, float, ())
        self.type = self.TAN_FOV
        self.fov = math.radians(30)

        minres = min(self.res)
        self.cx = self.res[0] / 2
        self.cy = self.res[1] / 2
        self.fx = minres / (2 * math.tan(self.fov))
        self.fy = minres / (2 * math.tan(self.fov))

        self.ctl = CameraCtl()

        @ti.materialize_callback
        @ti.kernel
        def init_intrinsic():
            # TODO: intrinsic as 3x3 projection matrix?
            self.intrinsic[None][0, 0] = self.fx
            self.intrinsic[None][0, 2] = self.cx
            self.intrinsic[None][1, 1] = self.fy
            self.intrinsic[None][1, 2] = self.cy
            self.intrinsic[None][2, 2] = 1.0

    def from_mouse(self, gui):
        changed = self.ctl.from_mouse(gui)
        self.ctl.apply(self)
        return changed

    @ti.func
    def render(self, scene):
        self.fb.clear_buffer()

        # sets up light directions
        if ti.static(len(scene.lights)):
            for light in ti.static(scene.lights):
                light.set_view(self)  # TODO: t3.Light should be a subclass of t3.ModelBase?
        else:
            ti.static_print('Warning: no lights')

        if ti.static(len(scene.models)):
            for model in ti.static(scene.models):
                model.set_view(self)  # sets up ModelView matrix
            for model in ti.static(scene.models):
                model.render(self)
        else:
            ti.static_print('Warning: no models')

    def set_intrinsic(self, fx=None, fy=None, cx=None, cy=None):
        # see http://ais.informatik.uni-freiburg.de/teaching/ws09/robotics2/pdfs/rob2-08-camera-calibration.pdf
        self.fx = fx or self.fx
        self.fy = fy or self.fy
        self.cx = cx or self.cx
        self.cy = cy or self.cy

    @property
    def img(self):
        return self.fb['img']

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


class CameraCtl:
    def __init__(self, pos=None, target=None, up=None):
        # python scope camera transformations
        self.pos = pos or [0, 0, -2]
        self.target = target or [0, 0, 0]
        self.up = up or [0, 1, 0]
        self.trans = None
        self.set()
        # mouse position for camera control
        self.mpos = (0, 0)

    def from_mouse(self, gui):
        changed = False
        is_alter_move = gui.is_pressed(ti.GUI.CTRL)
        if gui.is_pressed(ti.GUI.LMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.orbit((mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]),
                    pov=is_alter_move)
            self.mpos = mpos
            changed = True
        elif gui.is_pressed(ti.GUI.RMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.zoom_by_mouse(mpos, (mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]),
                        dolly=is_alter_move)
            self.mpos = mpos
            changed = True
        elif gui.is_pressed(ti.GUI.MMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.pan((mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]))
            self.mpos = mpos
            changed = True
        else:
            if gui.event and gui.event.key == ti.GUI.WHEEL:
                # one mouse wheel unit is (0, 120)
                self.zoom(-gui.event.delta[1] / 1200,
                    dolly=is_alter_move)
                gui.event = None
                changed = True
            mpos = (0, 0)
        self.mpos = mpos
        return changed

    '''
    NOTE: taichi_three uses a LEFT HANDED coordinate system.
    that is, the +Z axis points FROM the camera TOWARDS the scene,
    with X, Y being device coordinates.
    '''
    def set(self, pos=None, target=None, up=None):
        pos = ti.Vector(pos or self.pos)
        target = ti.Vector(target or self.target)
        up = ti.Vector(up or self.up)
        fwd = (target - pos).normalized()
        right = up.cross(fwd).normalized()
        up = fwd.cross(right)
        trans = ti.Matrix([right.entries, up.entries, fwd.entries]).transpose()

        self.trans = [[trans[i, j] for j in range(3)] for i in range(3)]
        self.pos = pos.entries
        self.target = target.entries

    def apply(self, camera):
        camera.L2W[None] = transform(self.trans, self.pos)

    def orbit(self, delta, sensitivity=2.75, pov=False):
        ds, dt = delta
        if ds != 0 or dt != 0:
            dis = math.sqrt(sum((self.target[i] - self.pos[i]) ** 2 for i in range(3)))
            ds, dt = ds * sensitivity, dt * sensitivity
            newdir = ts.vec3(ds, dt, 1).normalized()
            newdir = [sum(self.trans[i][j] * newdir[j] for j in range(3))\
                        for i in range(3)]
            if pov:
                newtarget = [self.pos[i] + dis * newdir[i] for i in range(3)]
                self.set(target=newtarget)
            else:
                newpos = [self.target[i] - dis * newdir[i] for i in range(3)]
                self.set(pos=newpos)

    def zoom_by_mouse(self, pos, delta, sensitivity=1.75, dolly=False):
        ds, dt = delta
        if ds != 0 or dt != 0:
            z = math.sqrt(ds ** 2 + dt ** 2) * sensitivity
            if (pos[0] - 0.5) * ds + (pos[1] - 0.5) * dt > 0:
                z *= -1
            self.zoom(z, dolly)
    
    def zoom(self, z, dolly=False):
        newpos = [(1 + z) * self.pos[i] - z * self.target[i] for i in range(3)]
        if dolly:
            newtarget = [z * self.pos[i] + (1 - z) * self.target[i] for i in range(3)]
            self.set(pos=newpos, target=newtarget)
        else:
            self.set(pos=newpos)

    def pan(self, delta, sensitivity=1.25):
        ds, dt = delta
        if ds != 0 or dt != 0:
            dis = math.sqrt(sum((self.target[i] - self.pos[i]) ** 2 for i in range(3)))
            ds, dt = ds * sensitivity, dt * sensitivity
            newdir = ts.vec3(-ds, -dt, 1).normalized()
            newdir = [sum(self.trans[i][j] * newdir[j] for j in range(3))\
                        for i in range(3)]
            newtarget = [self.pos[i] + dis * newdir[i] for i in range(3)]
            newpos = [self.pos[i] + newtarget[i] - self.target[i] for i in range(3)]
            self.set(pos=newpos, target=newtarget)