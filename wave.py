from tina.advans import *
from tina.path.geometry import *

ti.init(ti.opengl)


@ti.func
def blackbody(temp, wave):
    wave *= 1e-9
    HCC2 = 1.1910429723971884140794892e-29
    HKC = 1.438777085924334052222404423195819240925e-2
    return wave**-5 * HCC2 / (ti.exp(HKC / (wave * temp)) - 1)


@ti.func
def lam_to_rgb(lam):
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
def rgb_at_lam(rgb, lam):
    ret = 0.0
    if 380 <= lam < 490:
        ret += rgb.z / 3
    if 490 <= lam < 580:
        ret += rgb.y / 3
    if 580 <= lam <= 780:
        ret += rgb.x / 3
    return ret


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


res = 512, 512
img = ti.Vector.field(3, float, res)
IMVP = ti.Matrix.field(4, 4, float, ())
accum = tina.Accumator(res)

@ti.kernel
def render():
    for i, j in img:
        uv = V(i + ti.random(), j + ti.random()) / tovector(img.shape) * 2 - 1
        ro = mapply_pos(IMVP[None], V(uv.x, uv.y, -1.0))
        ro1 = mapply_pos(IMVP[None], V(uv.x, uv.y, 1.0))
        rd = (ro1 - ro).normalized()
        lam = randomLam()

        hit, depth = ray_sphere_hit(V(0., 0., 0.), 1., ro, rd)
        if hit:
            color = V(1., 1., 1.)
            radiance = rgb_at_lam(color, lam)
            img[i, j] = lam_to_rgb(lam) * radiance
        else:
            img[i, j] = 0.0



gui = ti.GUI(res=res)
ctrl = tina.Control(gui)
while gui.running:
    if ctrl.process_events():
        accum.clear()
    view, proj = ctrl.get_camera()
    IMVP[None] = np.linalg.inv(proj @ view).tolist()
    render()
    accum.update(img)
    gui.set_image(aces_tonemap(accum.img.to_numpy()))
    gui.show()
