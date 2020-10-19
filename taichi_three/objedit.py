import numpy as np
import math


def _t(x):
    return [[x[i][j] for i in range(len(x))] for j in range(len(x[0]))]


class Geometry:
    @classmethod
    def fromobjstr(cls, code, flip=False):
        from .loader import readobj
        from io import BytesIO
        obj = readobj(BytesIO(code))
        if flip:
            objflipface(obj)
        return obj

    @classmethod
    def fromarrays(cls, vertices, faces, texcoords, normals, flip=False):
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        texcoords = np.array(texcoords, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        obj = dict(vp=vertices, f=faces, vn=normals, vt=texcoords)
        objswapaxis(obj, 1, 2)
        if flip:
            objflipface(obj)
        return obj

    @classmethod
    def cylinder(cls, semiheight=1, N=32):
        from .loader import _tri_append
        vertices = [(0, 0, -semiheight), (0, 0, +semiheight)]
        texcoords = [(0, 0)]
        normals = [(0, 0, -1), (0, 0, 1)]
        faces = []
        for i in range(N):
            angle = (i / N) * np.pi * 2
            pos = math.cos(angle), math.sin(angle)
            vertices.append((*pos, -semiheight))
            vertices.append((*pos, +semiheight))
            normals.append((*pos, 0))
            texcoords.append(pos)
        for i in range(N):
            j = (i + 1) % N
            a, b = i * 2 + 2, j * 2 + 2
            c, d = j * 2 + 3, i * 2 + 3
            faces.append(_t([[0, a, b], [0, i, j], [0, 0, 0]]))
            faces.append(_t([[1, c, d], [0, j, i], [1, 1, 1]]))
            _tri_append(faces, _t([[d, c, b, a], [i, j, j, i], [i + 2, j + 2, j + 2, i + 2]]))
        return cls.fromarrays(vertices, faces, texcoords, normals, flip=True)

    @classmethod
    def meshgrid(cls, n):
        def _face(x, y):
            return np.array([(x, y), (x, y + 1), (x + 1, y + 1), (x + 1, y)])

        n_particles = n**2
        n_faces = (n - 1)**2
        xi = np.arange(n)
        yi = np.arange(n)
        xs = np.linspace(0, 1, n)
        ys = np.linspace(0, 1, n)
        uv = np.array(np.meshgrid(xs, ys)).swapaxes(0, 2).reshape(n_particles, 2)
        faces = _face(*np.meshgrid(xi[:-1], yi[:-1])).swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3)
        faces = (faces[1] * n + faces[0]).reshape(n_faces, 4)
        pos = np.concatenate([uv * 2 - 1, np.zeros((n_particles, 1))], axis=1)
        faces = np.moveaxis(np.array([faces, faces, np.zeros((n_faces, 4), dtype=np.int_)]), 0, 2)
        faces = np.concatenate([faces[:, (0, 1, 2)], faces[:, (0, 2, 3)]], axis=0)
        normals = np.array([[0, 0, 1]])
        return cls.fromarrays(pos, faces, uv, normals)

    @classmethod
    def cube(cls):
        return cls.fromobjstr(b'''o Cube
v 1.0 1.0 -1.0
v 1.0 -1.0 -1.0
v 1.0 1.0 1.0
v 1.0 -1.0 1.0
v -1.0 1.0 -1.0
v -1.0 -1.0 -1.0
v -1.0 1.0 1.0
v -1.0 -1.0 1.0
vt 0.0 0.0
vt 0.0 1.0
vt 1.0 1.0
vt 1.0 0.0
vn 0.0 1.0 0.0
vn 0.0 0.0 1.0
vn -1.0 0.0 0.0
vn 0.0 -1.0 0.0
vn 1.0 0.0 0.0
vn 0.0 0.0 -1.0
f 1/1/1 5/2/1 7/3/1
f 7/3/1 3/4/1 1/1/1
f 4/1/2 3/2/2 7/3/2
f 7/3/2 8/4/2 4/1/2
f 8/1/3 7/2/3 5/3/3
f 5/3/3 6/4/3 8/1/3
f 6/1/4 2/2/4 4/3/4
f 4/3/4 8/4/4 6/1/4
f 2/1/5 1/2/5 3/3/5
f 3/3/5 4/4/5 2/1/5
f 6/1/6 5/2/6 1/3/6
f 1/3/6 2/4/6 6/1/6
''')



def objunpackmtls(obj):
    faces = obj['f']
    parts = {}
    ends = []
    for end, name in obj['usemtl']:
        ends.append(end)
    ends.append(len(faces))
    ends.pop(0)
    for end, (beg, name) in zip(ends, obj['usemtl']):
        cur = {}
        cur['f'] = faces[beg:end]
        cur['vp'] = obj['vp']
        cur['vn'] = obj['vn']
        cur['vt'] = obj['vt']
        parts[name] = cur
    return parts


def objmerge(obj, other):
    obj['f'] = np.concatenate([obj['f'], other['f'] + len(obj['f'])], axis=0)
    obj['vp'] = np.concatenate([obj['vp'], other['vp']], axis=0)
    obj['vn'] = np.concatenate([obj['vn'], other['vn']], axis=0)
    obj['vt'] = np.concatenate([obj['vt'], other['vt']], axis=0)
    return obj


def objautoscale(obj):
    obj['vp'] -= np.average(obj['vp'], axis=0)
    obj['vp'] /= np.max(np.abs(obj['vp']))


def objflipaxis(obj, x=False, y=False, z=False):
    for i, flip in enumerate([x, y, z]):
        if flip:
            obj['vp'][:, i] = -obj['vp'][:, i]
            obj['vn'][:, i] = -obj['vn'][:, i]
    if (x != y) != z:
        objflipface(obj)


def objswapaxis(obj, a=1, b=2):
    obj['vp'][:, (a, b)] = obj['vp'][:, (b, a)]
    obj['vn'][:, (a, b)] = obj['vn'][:, (b, a)]


def objreorient(obj, orient):
    flip = False
    if orient.startswith('-'):
        flip = True
        orient = orient[1:]

    x, y, z = ['xyz'.index(o.lower()) for o in orient]
    fx, fy, fz = [o.isupper() for o in orient]

    if x != 0 or y != 1 or z != 2:
        obj['vp'][:, (0, 1, 2)] = obj['vp'][:, (x, y, z)]
        obj['vn'][:, (0, 1, 2)] = obj['vn'][:, (x, y, z)]

    for i, fi in enumerate([fx, fy, fz]):
        if fi:
            obj['vp'][:, i] = -obj['vp'][:, i]
            obj['vn'][:, i] = -obj['vn'][:, i]

    if flip:
        objflipface(obj)
    return obj


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
    nrm = np.cross(p[:, 2] - p[:, 0], p[:, 1] - p[:, 0])
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