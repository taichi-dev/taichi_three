from ..common import *


@ti.func
def blackbody(temp, wave):
    wave *= 1e-9
    HCC2 = 1.1910429723971884140794892e-29
    HKC = 1.438777085924334052222404423195819240925e-2
    return wave ** -5 * HCC2 / (ti.exp(HKC / (wave * temp)) - 1)


@ti.func
def wav_to_rgb(wav):
    # https://blog.csdn.net/tanmx219/article/details/91658415
    r = 0.0
    g = 0.0
    b = 0.0
    alpha = 0.0

    # 3, 3, 2.5
    if 380.0 <= wav < 440.0:
        # .5, 0, 1
        r = -1.0 * (wav - 440.0) / (440.0 - 380.0)
        g = 0.0
        b = 1.0
    elif 440.0 <= wav < 490.0:
        # 0, .5, 1
        r = 0.0
        g = (wav - 440.0) / (490.0 - 440.0)
        b = 1.0
    elif 490.0 <= wav < 510.0:
        # 0, 1, .5
        r = 0.0
        g = 1.0
        b = -1.0 * (wav - 510.0) / (510.0 - 490.0)
    elif 510.0 <= wav < 580.0:
        # .5, 1, 0
        r = (wav - 510.0) / (580.0 - 510.0)
        g = 1.0
        b = 0.0
    elif 580.0 <= wav < 645.0:
        # 1, .5, 0
        r = 1.0
        g = -1.0 * (wav - 645.0) / (645.0 - 580.0)
        b = 0.0
    elif 645.0 <= wav < 780.0:
        # 1, 0, 0
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r = 0.0
        g = 0.0
        b = 0.0

    # 在可见光谱的边缘处强度较低。
    if 380.0 <= wav < 420.0:
        alpha = 0.30 + 0.70 * (wav - 380.0) / (420.0 - 380.0)
    elif 420.0 <= wav < 701.0:
        alpha = 1.0
    elif 701.0 <= wav < 780.0:
        alpha = 0.30 + 0.70 * (780.0 - wav) / (780.0 - 700.0)
    else:
        alpha = 0.0

    r *= 1.0 / 0.43511572
    g *= 1.0 / 0.5000342
    b *= 3.0 / 2.5 / 0.45338875
    return V(r, g, b) * alpha


@ti.func
def rgb_at_wav(rgb, wav):
    if ti.static(not isinstance(rgb, ti.Matrix)):
        return rgb
    ret = 0.0
    r, g, b = rgb
    if 645 <= wav < 780:
        ret += r / (0.3041293 * 1.179588 * 1.0081447)
    if 500 <= wav < 540:
        ret += g / (0.3092271 * 1.0824629 * 0.9944099)
    if 420 <= wav < 460:
        ret += b / (0.3240771 * 1.1763208 * 1.0136887)
    return ret


@ti.func
def random_wav(cho):
    #cho = ti.random(int) % 6
    cho = cho % 6
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
