from ..common import *


print('[Tina] Hint: LMB to orbit, RMB to pan, wheel to zoom')


class Control:
    def __init__(self, gui, fov=60, blendish=False):
        self.gui = gui
        self.center = np.array([0, 0, 0], dtype=float)
        self.up = np.array([0, 1, 1e-12], dtype=float)
        self.back = np.array([0, 0, 1], dtype=float)
        self.dist = 3
        self.scale = 1.0
        self.fov = fov

        self.lmb = None
        self.mmb = None
        self.rmb = None
        self.blendish = blendish

    def process_events(self):
        return any([self.on_event(e) for e in self.gui.get_events()])

    def on_pan(self, delta, origin):
        right = np.cross(self.up, self.back)
        up = np.cross(self.back, right)

        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)

        delta *= 2
        self.center -= (right * delta[0] + up * delta[1]) / self.scale

    def on_orbit(self, delta, origin):
        delta_phi = delta[0] * ti.pi
        delta_theta = delta[1] * ti.pi
        pos = self.back
        radius = np.linalg.norm(pos)
        theta = np.arccos(pos[1] / radius)
        phi = np.arctan2(pos[2], pos[0])

        theta = np.clip(theta + delta_theta, 0, ti.pi)
        phi += delta_phi

        pos[0] = radius * np.sin(theta) * np.cos(phi)
        pos[2] = radius * np.sin(theta) * np.sin(phi)
        pos[1] = radius * np.cos(theta)

    def on_zoom(self, delta, origin):
        self.scale *= pow(1.12, delta)

    def on_lmb_drag(self, delta, origin):
        if not self.blendish:
            self.on_orbit(delta, origin)

    def on_mmb_drag(self, delta, origin):
        if self.blendish:
            if self.gui.is_pressed(self.gui.SHIFT):
                self.on_pan(delta, origin)
            else:
                self.on_orbit(delta, origin)

    def on_rmb_drag(self, delta, origin):
        if not self.blendish:
            self.on_pan(delta, origin)

    def on_wheel(self, delta, origin):
        self.on_zoom(delta, origin)

    def get_camera(self, engine):
        ret = self.process_events()

        from ..core.camera import lookat, orthogonal, perspective

        aspect = self.gui.res[0] / self.gui.res[1]
        if self.fov == 0:
            view = lookat(self.center, self.back, self.up, self.dist)
            proj = orthogonal(1 / self.scale, aspect)
        else:
            view = lookat(self.center, self.back, self.up, self.dist / self.scale)
            proj = perspective(self.fov, aspect)

        engine.set_camera(view, proj)
        return ret

    def on_event(self, e):
        if e.type == self.gui.PRESS:
            if e.key == self.gui.TAB:
                if self.fov == 0:
                    self.fov = 60
                else:
                    self.fov = 0
                return True
            elif e.key == self.gui.ESCAPE:
                self.gui.running = False
        if e.key == self.gui.LMB:
            if e.type == self.gui.PRESS:
                self.lmb = np.array(e.pos)
                return True
            else:
                self.lmb = None
        elif e.key == self.gui.MMB:
            if e.type == self.gui.PRESS:
                self.mmb = np.array(e.pos)
                return True
            else:
                self.mmb = None
        elif e.key == self.gui.RMB:
            if e.type == self.gui.PRESS:
                self.rmb = np.array(e.pos)
                return True
            else:
                self.rmb = None
        elif e.key == self.gui.MOVE:
            if self.lmb is not None:
                new_lmb = np.array(e.pos)
                delta_lmb = new_lmb - self.lmb
                self.on_lmb_drag(delta_lmb, self.lmb)
                self.lmb = new_lmb
                return True
            if self.mmb is not None:
                new_mmb = np.array(e.pos)
                delta_mmb = new_mmb - self.mmb
                self.on_mmb_drag(delta_mmb, self.mmb)
                self.mmb = new_mmb
                return True
            if self.rmb is not None:
                new_rmb = np.array(e.pos)
                delta_rmb = new_rmb - self.rmb
                self.on_rmb_drag(delta_rmb, self.rmb)
                self.rmb = new_rmb
                return True
        elif e.key == self.gui.WHEEL:
            delta = e.delta[1] / 120
            self.on_wheel(delta, np.array(e.pos))
            return True

        return False
