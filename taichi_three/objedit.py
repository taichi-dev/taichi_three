import numpy as np


def objflipaxis(obj, *flips):
    for i, flip in enumerate(flips):
        if flip:
            obj['vp'][:, i] = -obj['vp'][:, i]
            obj['vn'][:, i] = -obj['vn'][:, i]
    if (flips[0] != flips[1]) != flips[2]:
        objflipface(obj)


def objflipface(obj):
    obj['f'][:, ::-1, :] = obj['f'][:, :, :]


def objflipnorm(obj):
    obj['vn'] = -obj['vn']


def objbothface(obj):
    tmp = np.array(obj['f'])
    tmp[:, ::-1, :] = obj['f'][:, :, :]
    obj['f'] = np.concatenate([obj['f'], tmp])
    obj['vn'] = np.concatenate([obj['vn'], -obj['vn']])


def objmknorm(obj):
    fip = obj['f'][:, :, 0]
    fit = obj['f'][:, :, 1]
    p = obj['vp'][fip]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    fin = np.arange(obj['f'].shape[0])[:, np.newaxis]
    fin = np.concatenate([fin for i in range(3)], axis=1)
    newf = np.array([fip, fit, fin]).swapaxes(1, 2).swapaxes(0, 2)
    obj['vn'] = nrm
    obj['f'] = newf


def objshow(obj, visual='color', res=(512, 512), ortho=False, showball=False, lightdir=[0.4, -1.5, 0.8]):
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