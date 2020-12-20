import taichi as ti
import numpy as np
import tina


def main(filename, radius=0.05):
    ti.init(ti.gpu)

    pos = np.load(filename).astype(np.float32)
    scene = tina.Scene((1024, 768))
    pars = tina.SimpleParticles(maxpars=len(pos))
    scene.add_object(pars)

    gui = ti.GUI('particles', scene.res, fast_gui=True)
    pars.set_particles(pos)
    pars.set_particle_radii(np.ones(len(pos), dtype=np.float32) * radius)
    while gui.running:
        scene.input(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.show()


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
