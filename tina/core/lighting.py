from ..advans import *


@ti.data_oriented
class SkyboxLighting:
    def __init__(self):
        pass

    def load_skybox(self, path, precision=32, cubic=True):
        if isinstance(path, str) and path.endswith('.npz'):
            print('[Tina] Loading pre-cooked IBL map from', path)
            data = np.load(path, allow_pickle=True)
            self.skybox = tina.Skybox(data['env'])
            self.ibls = {}
            self.ibls[tina.Lambert] = tina.Skybox(data['diff'])
            self.ibls[tina.CookTorrance] = tuple(map(tina.Skybox, data['spec'])), texture_as_field(data['lut'])
        else:
            if hasattr(path, 'sample'):
                self.skybox = path
            elif path.endswith('.npy'):
                self.skybox = tina.Skybox(np.load(path))
            else:
                self.skybox = tina.Skybox(path, cubic=cubic)
            self.ibls = {}
            for mattype in [tina.CookTorrance, tina.Lambert, tina.Mirror]:
                self.ibls[mattype] = mattype.cook_for_ibl(self.skybox, precision)

    def save(self, path):
        spec = tuple(x.img.to_numpy() for x in self.ibls[tina.CookTorrance][0])
        lut = self.ibls[tina.CookTorrance][1].to_numpy()
        diff = self.ibls[tina.Lambert].img.to_numpy()
        env = self.skybox.img.to_numpy()
        np.savez(path, env=env, spec=spec, lut=lut, diff=diff)

    @ti.func
    def background(self, rd):
        return self.skybox.sample(rd)

    @ti.func
    def shade_color(self, material, pos, normal, viewdir):
        ibl = ti.static(self.ibls.get(type(material), self.ibls))
        return material.sample_ibl(ibl, viewdir, normal)


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
        return i

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
