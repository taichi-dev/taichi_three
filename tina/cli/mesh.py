import taichi as ti
import tina


def main(filename):
    ti.init(ti.gpu)

    obj = tina.readobj(filename, scale='auto')
    scene = tina.Scene((1024, 768), maxfaces=len(obj['f']), smoothing=True)
    model = tina.MeshModel(obj)
    scene.add_object(model)

    gui = ti.GUI('mesh', scene.res, fast_gui=True)
    while gui.running:
        scene.input(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.show()


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
