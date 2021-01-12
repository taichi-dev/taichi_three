import taichi as ti
import numpy as np

img = ti.imread('noise.png')
#img = ti.imread('/opt/cuda/samples/3_Imaging/imageDenoising/data/portrait_noise.bmp')
img = np.float32(img / 255)
w, h, chans = img.shape

src = ti.Vector.field(chans, float, (w, h))
dst = ti.Vector.field(chans, float, (w, h))

src.from_numpy(img)

noise = 1 / 0.32**2
lerp_c = 0.2
win_rad = 3
blk_rad = 3
win_area = (2 * win_rad + 1)**2
wei_thres = 0.02
lerp_thres = 0.79

@ti.kernel
def denoise():
    for x, y in src:
        cnt = 0.0
        wei = 0.0
        clr = ti.Vector([0.0, 0.0, 0.0])

        clr00 = src[x, y]
        for i, j in ti.ndrange((-win_rad, win_rad + 1), (-win_rad, win_rad + 1)):
            wij = 0.0

            clrij = src[x + i, y + j]
            disij = (clr00 - clrij).norm_sqr()

            wij = ti.exp(-(disij * noise + (i**2 + j**2) / win_area))
            cnt += 1 / win_area if wij > wei_thres else 0
            clr += clrij * wij
            wei += wij

        clr /= wei
        lerp_q = lerp_c if cnt > lerp_thres else 1 - lerp_c
        dst[x, y] = clr * (1 - lerp_q) + src[x, y] * lerp_q


denoise()
gui1 = ti.GUI('before', (w, h))
gui2 = ti.GUI('after', (w, h))
gui1.fps_limit = 30
gui2.fps_limit = 30
while gui1.running and gui2.running:
    gui1.running = not gui1.get_event(ti.GUI.ESCAPE)
    gui2.running = not gui2.get_event(ti.GUI.ESCAPE)
    gui1.set_image(src)
    gui2.set_image(dst)
    gui1.show()
    gui2.show()
