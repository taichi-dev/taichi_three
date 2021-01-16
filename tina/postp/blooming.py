from ..advans import *


@ti.data_oriented
class Blooming:
    def __init__(self, res):
        self.res = tovector(res)
        self.img = ti.Vector.field(3, float, self.res // 2)
        self.tmp = ti.Vector.field(3, float, self.res // 2)
        self.gwei = ti.field(float, 128)
        self.thresh = ti.field(float, ())
        self.factor = ti.field(float, ())
        self.scale = ti.field(float, ())
        self.sigma = ti.field(float, ())
        self.radius = ti.field(int, ())

        @ti.materialize_callback
        def init_params():
            self.thresh[None] = 1
            self.factor[None] = 1
            self.radius[None] = min(*self.res) // 16
            self.sigma[None] = 1
            self.scale[None] = 0.25
            self.init_gwei()

    @ti.kernel
    def init_gwei(self):
        sum = -1.0
        radius = self.radius[None]
        sigma = self.sigma[None]
        for i in range(radius + 1):
            x = sigma * i / radius
            y = ti.exp(-x**2)
            self.gwei[i] = y
            sum += y * 2
        for i in range(radius + 1):
            self.gwei[i] /= sum

    @ti.func
    def filter(self, x):
        t = max(0, x - self.thresh[None])
        t = 1 - 1 / (1 + self.scale[None] * t)
        return self.factor[None] * t

    @ti.kernel
    def apply(self, image: ti.template()):
        for I in ti.grouped(self.img):
            res = V(0., 0., 0.)
            for J in ti.grouped(ti.ndrange(2, 2)):
                res += self.filter(image[I * 2 + J])
            self.img[I] = res / 4
        for I in ti.grouped(self.img):
            res = self.img[I] * self.gwei[0]
            for i in range(1, self.radius[None] + 1):
                val = self.img[max(0, I.x - i), I.y]
                val += self.img[min(self.img.shape[0] - 1, I.x + i), I.y]
                res += val * self.gwei[i]
            self.tmp[I] = res
        for I in ti.grouped(self.img):
            res = self.tmp[I] * self.gwei[0]
            for i in range(1, self.radius[None] + 1):
                dir = V(0, 1)
                val = self.tmp[I.x, max(0, I.y - i)]
                val += self.tmp[I.x, min(self.img.shape[1] - 1, I.y + i)]
                res += val * self.gwei[i]
            self.img[I] = res
        for I in ti.grouped(image):
            image[I] += bilerp(self.img, I / 2)
