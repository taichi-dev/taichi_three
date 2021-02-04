from tina.advans import *

ti.init(ti.cpu)

PBR = 1

if PBR:
    roughness = tina.Param(float, initial=0.4)
    metallic = tina.Param(float, initial=1.0)
    material = tina.PBR(basecolor=[1, 1, 1], roughness=roughness, metallic=metallic, specular=0.0)
else:
    shineness = tina.Param(float, initial=10)
    material = tina.Phong(diffuse=[0, 0, 0], specular=[1, 1, 1], shineness=shineness)

iu = ti.field(float, ())

k = 1
n = 400
f = ti.Vector.field(3, float, (n * k, n))

@ti.kernel
def render():
    idir = spherical(iu[None], 0.)
    for i, j in f:
        ou, ov = 1 - (i + 0.5) / n, (j + 0.5) / n
        odir = spherical(ou, ov)
        brdf = material.brdf(V(0., 0., 1.), idir, odir)
        f[i, j] = brdf * ou

gui = ti.GUI('brdf', (n * k, n))
if PBR:
    roughness.make_slider(gui, 'roughness')
    metallic.make_slider(gui, 'metallic')
else:
    shineness.make_slider(gui, 'shineness', 1, 128, 1)
iu_slider = gui.slider('U', 0, 1, 0.01)
avg_label = gui.label('average')

while gui.running:
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
    render()
    im = f.to_numpy()
    avg_label.value = np.sum(im) / (3 * k * n**2)
    gui.set_image(1 - np.exp(-im))
    gui.show()
    iu[None] = iu_slider.value
