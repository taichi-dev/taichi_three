import numpy as np


def objmerge(obj, other):
    obj['f'] = np.concatenate([obj['f'], other['f'] + len(obj['f'])], axis=0)
    obj['vp'] = np.concatenate([obj['vp'], other['vp']], axis=0)
    obj['vn'] = np.concatenate([obj['vn'], other['vn']], axis=0)
    obj['vt'] = np.concatenate([obj['vt'], other['vt']], axis=0)
    return obj


def objautoscale(self):
    obj['vp'] -= np.average(obj['vp'], axis=0)
    obj['vp'] /= np.max(np.abs(obj['vp']))


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


def objbreakdown(obj):
    res = {'f': [], 'vp': [], 'vt': [], 'vn': []}
    lp, lt, ln = len(obj['vp']), len(obj['vt']), len(obj['vn'])
    for i in range(len(obj['f'])):
        faces_i = obj['f'][i].swapaxes(0, 1)
        pos = obj['vp'][faces_i[0]]
        tex = obj['vt'][faces_i[1]]
        nrm = obj['vn'][faces_i[2]]
        np0 = (pos[1] + pos[2]) / 2
        np1 = (pos[2] + pos[0]) / 2
        np2 = (pos[0] + pos[1]) / 2
        nt0 = (tex[1] + tex[2]) / 2
        nt1 = (tex[2] + tex[0]) / 2
        nt2 = (tex[0] + tex[1]) / 2
        nn0 = nrm[1] + nrm[2]
        nn1 = nrm[2] + nrm[0]
        nn2 = nrm[0] + nrm[1]
        nn0 /= np.linalg.norm(nn0, axis=0, keepdims=True)
        nn1 /= np.linalg.norm(nn1, axis=0, keepdims=True)
        nn2 /= np.linalg.norm(nn2, axis=0, keepdims=True)
        res['vp'] += [np0, np1, np2]
        res['vt'] += [nt0, nt1, nt2]
        res['vn'] += [nn0, nn1, nn2]
        res['f'].append(np.array([
            [faces_i[0, 0], lp+2, lp+1],
            [faces_i[1, 0], lt+2, lt+1],
            [faces_i[2, 0], ln+2, ln+1],
        ], dtype=np.int32))
        res['f'].append(np.array([
            [faces_i[0, 1], lp+0, lp+2],
            [faces_i[1, 1], lt+0, lt+2],
            [faces_i[2, 1], ln+0, ln+2],
        ], dtype=np.int32))
        res['f'].append(np.array([
            [faces_i[0, 2], lp+1, lp+0],
            [faces_i[1, 2], lt+1, lt+0],
            [faces_i[2, 2], ln+1, ln+0],
        ], dtype=np.int32))
        res['f'].append(np.array([
            [lp+0, lp+1, lp+2],
            [lt+0, lt+1, lt+2],
            [ln+0, ln+1, ln+2],
        ], dtype=np.int32))
        lp += 3
        lt += 3
        ln += 3
    obj['f'] = np.array(res['f']).swapaxes(1, 2)
    obj['vp'] = np.concatenate([obj['vp'], np.array(res['vp'])], axis=0)
    obj['vt'] = np.concatenate([obj['vt'], np.array(res['vt'])], axis=0)
    obj['vn'] = np.concatenate([obj['vn'], np.array(res['vn'])], axis=0)


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