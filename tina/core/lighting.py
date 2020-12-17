from ..common import *


@ti.data_oriented
class Lighting:
    def __init__(self, maxlights=16):
        self.light_dirs = ti.Vector.field(4, float, maxlights)
        self.light_colors = ti.Vector.field(3, float, maxlights)
        self.ambient = ti.Vector.field(3, float, ())
        self.nlights = ti.field(int, ())

        @ti.materialize_callback
        @ti.kernel
        def init_lights():
            self.nlights[None] = 1
            self.light_dirs[0] = [0, 0, 1, 0]
            for i in self.light_dirs:
                self.light_colors[i] = [1, 1, 1]

    def set_lights(self, light_dirs):
        self.nlights[None] = len(light_dirs)
        for i, (dir, color) in enumerate(light_dirs):
            self.light_dirs[i] = dir
            self.light_colors[i] = color

    def set_ambient_light(self, color):
        self.ambient[None] = np.array(color).tolist()

    @ti.func
    def get_lights_range(self):
        for i in range(self.nlights[None]):
            yield i

    @ti.func
    def get_light_data(self, l):
        return self.light_dirs[l], self.light_colors[l]

    @ti.func
    def get_ambient_light_color(self):
        return self.ambient[None]
