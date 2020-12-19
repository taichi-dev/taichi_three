import taichi as ti
import tina

ti.init(ti.gpu)

scene = tina.Scene(raster_cls=tina.ParticleRaster)

gui = ti.GUI('particles')

while gui.running:
    scene.input(gui)
    scene.render()
    gui.set_image(scene.img)
    gui.show()
