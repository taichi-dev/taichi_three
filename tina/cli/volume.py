import taichi as ti
import numpy as np
import tina


def main(filename, density=16):
    ti.init(ti.gpu)

    dens = np.load(filename).astype(np.float32)
    scene = tina.Scene((1024, 768), N=dens.shape[0], taa=True, density=density)
    volume = tina.SimpleVolume(N=dens.shape[0])
    scene.add_object(volume)

    gui = ti.GUI('volume', scene.res, fast_gui=True)
    volume.set_volume_density(dens)
    while gui.running:
        scene.input(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.show()



if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
