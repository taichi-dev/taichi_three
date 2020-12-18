import taichi as ti
import tina


def main(*args):
    ti.init(ti.gpu)

    obj = tina.readobj(args[0], scale='auto')
    verts = obj['v'][obj['f'][:, :, 0]]
    norms = obj['vn'][obj['f'][:, :, 2]]

    engine = tina.Engine((1024, 768), maxfaces=len(verts), smoothing=True)

    img = ti.Vector.field(3, float, engine.res)
    shader = tina.SimpleShader(img)

    gui = ti.GUI('mesh visualize', engine.res, fast_gui=True)
    control = tina.Control(gui)

    accum = tina.Accumator(engine.res)

    engine.set_face_verts(verts)
    engine.set_face_norms(norms)

    while gui.running:
        engine.randomize_bias(accum.count[None] <= 1)
        if control.get_camera(engine):
            accum.clear()

        img.fill(0)
        engine.clear_depth()

        engine.render(shader)

        accum.update(img)
        gui.set_image(accum.img)
        gui.show()


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
