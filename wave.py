from tina.advans import *


@ti.func
def blackbody(temp, wave):
    wave *= 1e-9
    HCC2 = 1.1910429723971884140794892e-29
    HKC = 1.438777085924334052222404423195819240925e-2
    return wave**-5 * HCC2 / (ti.exp(HKC / (wave * temp)) - 1)


@ti.func
def lamToColor(lam):
    # https://blog.csdn.net/tanmx219/article/details/91658415
    r = 0.0
    g = 0.0
    b = 0.0
    alpha = 0.0

    #     3, 3, 2.5
    if (lam >= 380.0 and lam < 440.0):
        # .5, 0, 1
        r = -1.0 * (lam - 440.0) / (440.0 - 380.0);
        g = 0.0;
        b = 1.0;
    elif(lam >= 440.0 and lam < 490.0):
        # 0, .5, 1
        r = 0.0;
        g = (lam - 440.0) / (490.0 - 440.0);
        b = 1.0;
    elif(lam >= 490.0 and lam < 510.0):
        # 0, 1, .5
        r = 0.0;
        g = 1.0;
        b = -1.0 * (lam - 510.0) / (510.0 - 490.0);
    elif(lam >= 510.0 and lam < 580.0):
        # .5, 1, 0
        r = (lam - 510.0) / (580.0 - 510.0);
        g = 1.0;
        b = 0.0;
    elif(lam >= 580.0 and lam < 645.0):
        # 1, .5, 0
        r = 1.0;
        g = -1.0 * (lam - 645.0) / (645.0 - 580.0);
        b = 0.0;
    elif(lam >= 645.0 and lam <= 780.0):
        # 1, 0, 0
        r = 1.0;
        g = 0.0;
        b = 0.0;
    else:
        r = 0.0;
        g = 0.0;
        b = 0.0;

    #在可见光谱的边缘处强度较低。
    if (lam >= 380.0 and lam < 420.0):
        alpha = 0.30 + 0.70 * (lam - 380.0) / (420.0 - 380.0);
    elif(lam >= 420.0 and lam < 701.0):
        alpha = 1.0;
    elif(lam >= 701.0 and lam < 780.0):
        alpha = 0.30 + 0.70 * (780.0 - lam) / (780.0 - 700.0);
    else:
        alpha = 0.0;

    r *= 1.0 / 0.43511572
    g *= 1.0 / 0.5000342
    b *= 3.0 / 2.5 / 0.45338875
    return V(r, g, b) * alpha


@ti.func
def randomLam():
    cho = ti.random(int) % 6
    ret = ti.random()
    if cho == 0:
        ret = lerp(ret, 645, 780)
    elif cho == 1:
        ret = lerp(ret, 580, 645)
    elif cho == 2:
        ret = lerp(ret, 510, 580)
    elif cho == 3:
        ret = lerp(ret, 490, 510)
    elif cho == 4:
        ret = lerp(ret, 440, 490)
    elif cho == 5:
        ret = lerp(ret, 380, 440)
    return ret



res = 512, 128
img = ti.Vector.field(3, float, res)
accum = tina.Accumator(res)

@ti.kernel
def render():
    for i, j in img:
        lam = randomLam()
        radiance = blackbody(lerp(i / res[0], 1200, 3000), lam) * 1024
        img[i, j] = lamToColor(lam) * radiance


for i in range(32):
    render()
    accum.update(img)
ti.imshow(aces_tonemap(accum.img.to_numpy()))
