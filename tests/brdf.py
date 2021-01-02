from tina.advans import *

ti.init(ti.cpu)

metallic = tina.Param(float, initial=1.0)
roughness = tina.Param(float, initial=0.4)
material = tina.CookTorrance(basecolor=[1, 1, 1], metallic=metallic, roughness=roughness)

iu = ti.field(float, ())
iv = ti.field(float, ())

f = ti.Vector.field(3, float, (512, 512))

@ti.kernel
def render():
    idir = spherical(iu[None], iv[None])
    for i, j in f:
        ou, ov = i / 512, j / 512
        odir = spherical(ou, ov)
        f[i, j] = material.brdf(V(0., 0., 1.), idir, odir)

gui = ti.GUI('brdf')
metallic.make_slider(gui, 'metallic')
roughness.make_slider(gui, 'roughness')
iu_slider = gui.slider('U', 0, 1, 0.01)
iv_slider = gui.slider('V', 0, 1, 0.01)
avg_label = gui.label('average')

while gui.running:
    for e in gui.get_events():
        if e.key == gui.ESCAPE:
            gui.running = False
    render()
    im = f.to_numpy()
    avg_label.value = np.sum(im) / 512**2
    gui.set_image(1 - np.exp(-im))
    gui.show()
    iu[None] = iu_slider.value
    iv[None] = iv_slider.value
