import numpy as np


def objflip(obj, *flips):
    for i, flip in enumerate(flips):
        if flip:
            obj['vp'][:, i] = -obj['vp'][:, i]
            obj['vn'][:, i] = -obj['vn'][:, i]
    if (flips[0] != flips[1]) != flips[2]:  # FIXME
        obj['f'] = np.roll(obj['f'], 1, axis=1)


def objmknorm(obj):
    nrm = np.zeros((len(obj['f']), 3), dtype=np.float32)
    newf = np.zeros((len(obj['f']), len(obj['f'][0]), 3), dtype=np.int32)
    for i in range(len(obj['f'])):
        fip = obj['f'][i, :, 0]
        fit = obj['f'][i, :, 1]
        p = obj['vp'][fip]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        n /= np.sqrt(np.sum(n**2))
        nrm[i] = n
        fin = np.array([i, i, i])
        newf[i] = np.array([fip, fit, fin]).swapaxes(0, 1)
    obj['vn'] = nrm
    obj['f'] = newf


def objshow(obj, visual='color', res=(512, 512), ortho=False, showball=True, lightdir=[0.4, -1.5, 0.8]):
    import taichi_three as t3

    t3.reset()
    scene = t3.Scene()
    model = t3.Model.from_obj(obj)
    scene.add_model(model)
    if showball:
        ball = t3.Model.from_obj(t3.readobj('assets/sphere.obj', scale=0.6))
        scene.add_model(ball)
    camera = t3.Camera(res=res)
    if visual != 'color':
        dim = 3
        if visual == 'idepth':
            dim = 0
        if visual == 'texcoor':
            dim = 2
        camera.fb.add_buffer('normal', dim)
    if ortho:
        camera.type = camera.ORTHO
    scene.add_camera(camera)
    light = t3.Light(dir=lightdir)
    scene.add_light(light)

    gui = t3.GUI('Model', camera.res)
    while gui.running:
        gui.get_event(None)
        gui.running = not gui.is_pressed(gui.ESCAPE)
        camera.from_mouse(gui)
        if showball:
            ball.L2W.offset[None] = t3.Vector([1.75, -1.75, 0.0])
        scene.render()
        if visual == 'normal':
            gui.set_image(camera.fb['normal'].to_numpy() * 0.5 + 0.5)
        elif visual == 'color':
            gui.set_image(camera.img)
        else:
            gui.set_image(camera.fb[visual].to_numpy())
        gui.show()


if __name__ == '__main__':
    import taichi_three as t3

    obj = t3.readobj('assets/sphere.obj')
    t3.objmknorm(obj)
    t3.objshow(obj, 'normal')