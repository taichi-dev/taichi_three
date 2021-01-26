from ..advans import *


@ti.data_oriented
class SkyboxLighting:
    def __init__(self, skybox, ibltab=None):
        self.skybox = skybox
        if ibltab is None:
            precision = 32
            ibltab = {'env': skybox}
            tina.Lambert.cook_for_ibl(ibltab, precision)
            tina.CookTorrance.cook_for_ibl(ibltab, precision)
        self.ibltab = ibltab

    @ti.func
    def background(self, rd):
        return self.skybox.sample(rd)

    @ti.func
    def shade_color(self, material, pos, normal, viewdir):
        return material.sample_ibl(self.ibltab, viewdir, normal) + material.emission()


@ti.data_oriented
class Lighting:
    def __init__(self, maxlights=16):
        self.light_dirs = ti.Vector.field(4, float, maxlights)
        self.light_colors = ti.Vector.field(3, float, maxlights)
        self.ambient_color = ti.Vector.field(3, float, ())
        self.nlights = ti.field(int, ())

        @ti.materialize_callback
        @ti.kernel
        def init_lights():
            self.nlights[None] = 0
            for i in self.light_dirs:
                self.light_dirs[i] = [0, 0, 1, 0]
                self.light_colors[i] = [1, 1, 1]

    def set_lights(self, light_dirs):
        self.nlights[None] = len(light_dirs)
        for i, (dir, color) in enumerate(light_dirs):
            self.light_dirs[i] = dir
            self.light_colors[i] = color

    def clear_lights(self):
        self.nlights[None] = 0

    def add_light(self, dir=(0, 0, 1), pos=None, color=(1, 1, 1)):
        i = self.nlights[None]
        self.nlights[None] = i + 1
        self.set_light(i, dir, pos, color)
        return i

    def set_light(self, i, dir=(0, 0, 1), pos=None, color=(1, 1, 1)):
        if pos is not None:
            dir = np.array(pos)
            dirw = 1
        else:
            dir = np.array(dir)
            dir = dir / np.linalg.norm(dir)
            dirw = 0
        color = np.array(color)
        self.light_dirs[i] = dir.tolist() + [dirw]
        self.light_colors[i] = color.tolist()

    def set_ambient_light(self, color):
        self.ambient_color[None] = np.array(color).tolist()

    @ti.func
    def get_lights_range(self):
        for i in range(self.nlights[None]):
            yield i

    @ti.func
    def get_light_data(self, l):
        return self.light_dirs[l], self.light_colors[l]

    @ti.func
    def get_ambient_light_color(self):
        return self.ambient_color[None]

    @ti.func
    def shade_color(self, material, pos, normal, viewdir):
        res = V(0.0, 0.0, 0.0)
        res += material.emission()
        res += self.get_ambient_light_color() * material.ambient()
        for l in ti.smart(self.get_lights_range()):
            light, lcolor = self.get_light_data(l)
            light_dir = light.xyz - pos * light.w
            light_distance = light_dir.norm()
            light_dir /= light_distance
            cos_i = normal.dot(light_dir)
            if cos_i > 0:
                lcolor /= light_distance**2
                mcolor = material.brdf(normal, light_dir, viewdir)
                res += cos_i * lcolor * mcolor
        return res
