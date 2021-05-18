from ..common import *
import math

print('[Tina] Hint: MMB to orbit, Shift+MMB to pan, wheel to zoom')


class Control:
    def __init__(self, gui, fov=60, is_ortho=False, blendish=True):
        self.gui = gui
        self.center = np.array([0, 0, 0], dtype=float)
        #self.up = np.array([0, 1, 1e-12], dtype=float)
        self.is_ortho = is_ortho
        self.radius = 3.0
        self.R = np.eye(4)
        self.fov = fov

        self.last_mouse = None
        self.blendish = blendish

    def init_rot(self, theta, phi):
        from transformations import euler_matrix

        if theta == None:
            theta = 0
        if phi == None:
            phi = 0
        self.R = euler_matrix(-theta, phi, 0, 'sxyz')
        
    def process_events(self):
        ret = False
        for e in self.gui.get_events():
            if self.on_event(e):
                ret = True
        if self.check_mouse_move():
            ret = True
        return ret

    def on_pan(self, delta, origin):
        vx = -delta[0] * np.pi
        vy = -delta[1] * np.pi
        vz = 0
        v = np.array([vx, vy, vz])
        v = self.R[0:3, 0:3].dot(v)*0.5
        self.center += v * self.radius

    def on_orbit(self, delta, origin):
        from .matrix import RotationStep
        wx = delta[1] * np.pi
        wy = -delta[0] * np.pi
        wz = 0
        self.R = RotationStep(self.R, wx, wy, wz)

    @property
    def back(self):
        b = np.array([0, 0, self.radius], dtype=float)
        b = self.R[0:3,0:3].dot(b)
        return b

    def on_zoom(self, delta, origin):
        self.radius *= pow(0.89, delta)

    def on_fovadj(self, delta, origin):
        self.radius *= pow(0.89, delta)
        self.fov *= pow(0.89, -delta)

    def on_lmb_drag(self, delta, origin):
        if not self.blendish:
            if self.gui.is_pressed(self.gui.CTRL):
                delta = delta * 0.2
            if self.gui.is_pressed(self.gui.SHIFT):
                delta = delta * 5
            self.on_orbit(delta, origin)

    def on_mmb_drag(self, delta, origin):
        if self.blendish:
            if self.gui.is_pressed(self.gui.SHIFT):
                self.on_pan(delta, origin)
            else:
                self.on_orbit(delta, origin)

    def on_rmb_drag(self, delta, origin):
        if not self.blendish:
            if self.gui.is_pressed(self.gui.CTRL):
                delta = delta * 0.2
            if self.gui.is_pressed(self.gui.SHIFT):
                delta = delta * 5
            self.on_pan(delta, origin)

    def on_wheel(self, delta, origin):
        if not self.blendish:
            if self.gui.is_pressed(self.gui.CTRL):
                self.on_fovadj(delta, origin)
            else:
                if self.gui.is_pressed(self.gui.CTRL):
                    delta = delta * 0.2
                if self.gui.is_pressed(self.gui.SHIFT):
                    delta = delta * 5
                self.on_zoom(delta, origin)
        else:
            self.on_zoom(delta, origin)

    def get_camera(self):
        from .matrix import lookat, orthogonal, perspective, affine

        aspect = self.gui.res[0] / self.gui.res[1]
        if self.is_ortho:
            view = lookat(self.center, self.back / self.radius, self.up)
            proj = orthogonal(self.radius, aspect)
        else:
            view = np.linalg.inv(affine(self.R[0:3,0:3], (self.center + self.back)))
            proj = perspective(self.fov, aspect)

        return view, proj

    def apply_camera(self, engine):
        ret = self.process_events()
        view, proj = self.get_camera()
        engine.set_camera(view, proj)
        return ret

    def on_event(self, e):
        if e.type == self.gui.PRESS:
            if e.key == self.gui.TAB:
                self.is_ortho = not self.is_ortho
                return True

            elif e.key == self.gui.ESCAPE:
                self.gui.running = False

        elif e.type == self.gui.MOTION:
            if e.key == self.gui.WHEEL:
                delta = e.delta[1] / 120
                self.on_wheel(delta, np.array(e.pos))
                return True

        return False

    def check_mouse_move(self):
        ret = False
        curr_mouse = np.array(self.gui.get_cursor_pos())
        btn = list(map(self.gui.is_pressed, [self.gui.LMB, self.gui.MMB, self.gui.RMB]))
        if self.last_mouse is not None:
            delta = curr_mouse - self.last_mouse
            if delta[0] or delta[1]:
                if btn[0]:
                    self.on_lmb_drag(delta, self.last_mouse)
                if btn[1]:
                    self.on_mmb_drag(delta, self.last_mouse)
                if btn[2]:
                    self.on_rmb_drag(delta, self.last_mouse)
                ret = any(btn)

        if any(btn):
            self.last_mouse = curr_mouse
        else:
            self.last_mouse = None
        return ret
