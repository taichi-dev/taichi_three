from ..common import *


@ti.data_oriented
class Denoise:
    def __init__(self, res):
        self.res = tovector(res)

        self.src = ti.Vector.field(3, float, self.res)
        self.dst = ti.Vector.field(3, float, self.res)

    def knn(self, radius=3, noiseness=0.32, wei_thres=0.02,
            lerp_thres=0.79, lerp_factor=0.2):
        self._knn(radius, noiseness, wei_thres,
                lerp_thres, lerp_factor)

    def nlm(self, radius=3, noiseness=1.45, wei_thres=0.1,
            lerp_thres=0.1, lerp_factor=0.2):
        self._nlm(radius, noiseness, wei_thres,
                lerp_thres, lerp_factor)

    @ti.kernel
    def _knn(self, radius: int, noiseness: float, wei_thres: float,
            lerp_thres: float, lerp_factor: float):
        for x, y in self.src:
            cnt = 0.0
            wei = 0.0
            clr = ti.zero(self.src[x, y])

            noise = 1 / max(1e-5, noiseness**2)
            inv_area = 1 / (2 * radius + 1)**2

            clr00 = self.src[x, y]
            for i, j in ti.ndrange((-radius, radius + 1), (-radius, radius + 1)):
                clrij = self.src[x + i, y + j]
                disij = Vlen2(clr00 - clrij)

                wij = ti.exp(-(disij * noise + (i**2 + j**2) * inv_area))
                cnt += inv_area if wij > wei_thres else 0
                clr += clrij * wij
                wei += wij

            clr /= wei
            lerp_q = lerp_factor if cnt > lerp_thres else 1 - lerp_factor
            self.dst[x, y] = clr * (1 - lerp_q) + clr00 * lerp_q

    @ti.kernel
    def _nlm(self, radius: int, noiseness: float, wei_thres: float,
            lerp_thres: float, lerp_factor: float):
        for x, y in self.src:
            cnt = 0.0
            wei = 0.0
            clr = ti.zero(self.src[x, y])

            noise = 1 / max(1e-5, noiseness**2)
            inv_area = 1 / (2 * radius + 1)**2

            for i, j in ti.ndrange((-radius, radius + 1), (-radius, radius + 1)):
                wij = 0.0
                for m, n in ti.ndrange((-radius, radius + 1), (-radius, radius + 1)):
                    clr00 = self.src[x + m, y + n]
                    clrij = self.src[x + i + m, y + j + n]
                    wij += Vlen2(clr00 - clrij)

                wij = ti.exp(-(wij * noise + (i**2 + j**2) * inv_area))
                cnt += inv_area if wij > wei_thres else 0
                clr += self.src[x + i, y + j] * wij
                wei += wij

            clr /= wei
            lerp_q = lerp_factor if cnt > lerp_thres else 1 - lerp_factor
            self.dst[x, y] = clr * (1 - lerp_q) + self.src[x, y] * lerp_q


if __name__ == '__main__':
    ti.init(ti.gpu)

    img = ti.imread('noise.png')
    #img = ti.imread('cornell.png')
    #img = ti.imread('/opt/cuda/samples/3_Imaging/imageDenoising/data/portrait_noise.bmp')
    img = np.float32(img / 255)
    w, h, _ = img.shape

    denoise = Denoise((w, h))
    denoise.src.from_numpy(img)

    gui = ti.GUI('denoise', (w, h))

    knn = 0
    if knn:
        radius = gui.slider('radius', 0, 6, 1)
        noiseness = gui.slider('noiseness', 0, 3, 0.01)
        wei_thres = gui.slider('wei_thres', 0, 1, 0.01)
        lerp_thres = gui.slider('lerp_thres', 0, 3, 0.01)
        lerp_factor = gui.slider('lerp_factor', 0, 1, 0.01)
        radius.value = 3
        noiseness.value = 0.32
        wei_thres.value = 0.02
        lerp_thres.value = 0.79
        lerp_factor.value = 0.2
    else:
        radius = gui.slider('radius', 0, 6, 1)
        noiseness = gui.slider('noiseness', 0, 5, 0.01)
        wei_thres = gui.slider('wei_thres', 0, 1, 0.01)
        lerp_thres = gui.slider('lerp_thres', 0, 1, 0.01)
        lerp_factor = gui.slider('lerp_factor', 0, 1, 0.01)
        radius.value = 3
        noiseness.value = 1.45
        wei_thres.value = 0.1
        lerp_thres.value = 0.1
        lerp_factor.value = 0.2

    denoise.nlm()
    while gui.running:
        gui.running = not gui.get_event(gui.ESCAPE)
        if knn:
            denoise.knn(int(radius.value), noiseness.value, wei_thres.value, lerp_thres.value, lerp_factor.value)
        else:
            denoise.nlm(int(radius.value), noiseness.value, wei_thres.value, lerp_thres.value, lerp_factor.value)
        gui.set_image(denoise.dst)
        gui.show()
